#pragma once

#include <cassert>
#include <cstring>
#include <atomic>
#include <immintrin.h>
#include <sched.h>
#include <algorithm>

namespace cpsbtreeolc {

enum class PageType : uint8_t { BTreeInner=1, BTreeLeaf=2 };

static const uint8_t POINTER_SIZE = 8;
static const int MaxEntries = 100;

struct OptLock {
  std::atomic<uint64_t> typeVersionLockObsolete{0b100};

  bool isLocked(uint64_t version) {
    return ((version & 0b10) == 0b10);
  }

  uint64_t readLockOrRestart(bool &needRestart) {
    uint64_t version;
    version = typeVersionLockObsolete.load();
    if (isLocked(version) || isObsolete(version)) {
      _mm_pause();
      needRestart = true;
    }
    return version;
  }

  void writeLockOrRestart(bool &needRestart) {
    uint64_t version;
    version = readLockOrRestart(needRestart);
    if (needRestart) return;

    upgradeToWriteLockOrRestart(version, needRestart);
    if (needRestart) return;
  }

  void upgradeToWriteLockOrRestart(uint64_t &version, bool &needRestart) {
    if (typeVersionLockObsolete.compare_exchange_strong(version, version + 0b10)) {
      version = version + 0b10;
    } else {
      _mm_pause();
      needRestart = true;
    }
  }

  void writeUnlock() {
    typeVersionLockObsolete.fetch_add(0b10);
  }

  bool isObsolete(uint64_t version) {
    return (version & 1) == 1;
  }

  void checkOrRestart(uint64_t startRead, bool &needRestart) const {
    readUnlockOrRestart(startRead, needRestart);
  }

  void readUnlockOrRestart(uint64_t startRead, bool &needRestart) const {
    needRestart = (startRead != typeVersionLockObsolete.load());
  }

  void writeUnlockObsolete() {
    typeVersionLockObsolete.fetch_add(0b11);
  }
};

struct NodeBase : public OptLock{
  PageType type;
  uint16_t count;
};

struct BTreeLeafBase : public NodeBase {
  static const PageType typeMarker=PageType::BTreeLeaf;
};

class Key {
 private:
  // if prefix is too long
  // key stores the pointer to the prefix
  char key[POINTER_SIZE];
  uint16_t part_len_;

 public:
  Key() {
    part_len_ = 0;
    memset(key, 0, POINTER_SIZE);
  }

  Key(Key &right) {
    part_len_ = 0;
    memset(key, 0, POINTER_SIZE);
    setKeyStr(right.getKeyStr(), right.getLen());
  }

  Key &operator = (Key &right) {
    part_len_ = 0;
    memset(key, 0, POINTER_SIZE);
    setKeyStr(right.getKeyStr(), right.getLen());
    return *this;
  }

  ~Key() {
    if (isOverFlow()) {
      const char *overflow_str = getOverFlowStr();
//      std::cout << overflow_str << std::endl;
      delete [](getOverFlowStr());
    }
  }

  void setOverFlow() {
    uint16_t mask = 1 << (sizeof(uint16_t) * 8 - 1);
    part_len_ |= mask;
  }

  void clearOverFlow() {
    uint16_t mask = (1 << (sizeof(uint16_t) * 8 - 1)) - 1;
    part_len_ &= mask;
  }

  bool isOverFlow() const {
    uint16_t mask = 1 << (sizeof(uint16_t) * 8 - 1);
    return (part_len_ & mask) > 0;
  }

  const char *getOverFlowStr(){
    return *reinterpret_cast<const char **>(key);
  }

  uint16_t getLen() {
    uint16_t mask = (1 << (sizeof(uint16_t) * 8 - 1)) - 1;
    return part_len_ & mask;
  }

  void setLen(uint16_t new_len) {
    part_len_ = new_len;
    if (new_len > POINTER_SIZE)
      setOverFlow();
  }

  int64_t getSize() { // in bytes
    int64_t re = POINTER_SIZE + sizeof(uint8_t);
    if (isOverFlow())
      re += getLen();
    return re;
  }

  void setLength(uint16_t new_len) {
    if (getLen() > POINTER_SIZE && new_len <= POINTER_SIZE) {
      const char *overflow_str = getOverFlowStr();
      memmove(key, overflow_str, new_len);
      delete []overflow_str;
    }
    setLen(new_len);
  }

  void setKeyStr(const char *str, uint16_t len) {
    assert(getLen() < (1<<15));
    if (isOverFlow()) delete [](getOverFlowStr());

    if (len > POINTER_SIZE) {
      char *overflow_key = new char[len];
      memcpy(overflow_key, str, len);
      memcpy(key, &overflow_key, POINTER_SIZE);
    } else {
      for (int i = 0; i < len; i++) {
        key[i] = str[i];
      }
    }
    setLen(len);
  }

  const char *getKeyStr(){
    if (getLen() > POINTER_SIZE) {
      return getOverFlowStr();
    }
    return &key[0];
  }

  int compare(Key &right, Key &prefix) {
    uint16_t mylen = getLen();
    char *my_str = new char[mylen + 1];
    memcpy(my_str, getKeyStr(), mylen);
    my_str[mylen] = '\0';

    const char *prefix_str = prefix.getKeyStr();
    const char *right_str = right.getKeyStr();

    // Concatenate to get the full key
    char *full_key = new char[prefix.getLen() + right.getLen() + 1];
    uint16_t prefix_len = prefix.getLen();
    uint16_t right_len = right.getLen();
    memcpy(full_key, prefix_str, prefix_len);
    memcpy(full_key + prefix_len, right_str, right_len);
    uint16_t full_key_len = prefix_len + right_len;
    full_key[full_key_len] = '\0';

    int cmp = strcmp(my_str, full_key);
    delete []full_key;
    return cmp;
  }

  int commonPrefix(Key &right) {
    uint16_t len = std::min(getLen(), right.getLen());
    const char *my_str = getKeyStr();
    const char *right_str = right.getKeyStr();
    int i = 0;
    while (i < len) {
      if (my_str[i] == right_str[i])
        i++;
      else
        break;
    }
    return i;
  }

  void addHead(Key &prefix) {
    uint16_t prefix_key_len = prefix.getLen();
    uint16_t cur_len = getLen();
    char new_head_char = prefix.getKeyStr()[prefix_key_len - 1];
    // already overflow
    if (cur_len > POINTER_SIZE) {
      char *overflow_key = *reinterpret_cast<char **>(key);
      memmove(overflow_key + 1, overflow_key, cur_len);
      overflow_key[0] = new_head_char;
    } else if (cur_len == POINTER_SIZE) {
      // overflow
      char *overflow_key = new char[POINTER_SIZE + 1];
      overflow_key[0] = new_head_char;
      memcpy(overflow_key + 1, key, POINTER_SIZE);
      memcpy(key, &overflow_key, POINTER_SIZE);
    } else {
      // no overflow
      memmove(key + 1, key, cur_len);
      key[0] = new_head_char;
    }
    setLen(cur_len + 1);
  }

  void removeHead() {
    uint16_t key_len = getLen();
    assert(key_len > 0);
    if (key_len > POINTER_SIZE + 1) {
      char *overflow_key = *reinterpret_cast<char **>(key);
      memmove(overflow_key, overflow_key + 1, key_len - 1);
    } else if (key_len == POINTER_SIZE + 1) {
      char *overflow_key = *reinterpret_cast<char **>(key);
      memmove(key, overflow_key + 1, POINTER_SIZE);
      delete []overflow_key;
    } else {
      memmove(key, key + 1, key_len - 1);
    }
    setLen(key_len - 1);
  }

  // remove the head number of bytes
  void chunkBeginning(uint16_t cnt) {
    uint16_t key_len = getLen();
    assert(cnt <= key_len);
    for (int i = 0; i < cnt; i++) {
      removeHead();
    }
  }
};

template<class Payload>
struct BTreeLeaf : public BTreeLeafBase {

  Key prefix_key_;
  Key keys[MaxEntries];
  Payload payloads[MaxEntries];

  BTreeLeaf() {
    count=0;
    type=typeMarker;
  }

  bool isFull() { return count==MaxEntries; };

  int64_t getSize() {
    int64_t key_size = prefix_key_.getSize();
    return key_size + sizeof(Payload) * MaxEntries + key_size * MaxEntries;
  }

  unsigned lowerBound(Key &k) {
    unsigned lower=0;
    unsigned upper=count;
    do {
      unsigned mid=((upper-lower)/2)+lower;
      Key mid_key = keys[mid];
//      std::cout << "Mid key 0" << keys[0].getKeyStr() << std::endl;
//      std::cout << "Mid key 1" << keys[1].getKeyStr() << std::endl;
//      std::cout << "Mid key 2" << mid_key.getKeyStr() << std::endl;

      int cmp = k.compare(mid_key, prefix_key_);
      if (cmp < 0) {
        upper = mid;
      } else if (cmp > 0) {
        lower=mid + 1;
      } else {
        return mid;
      }
    } while (lower < upper);
    return lower;
  }

  unsigned lowerBoundBF(Key k) {
    auto base=keys;
    unsigned n=count;
    while (n>1) {
      const unsigned half=n/2;
      base=(base[half]<k)?(base+half):base;
      n-=half;
    }
    return (*base<k)+base-keys;
  }

  void insert(Key k,Payload p) {
    assert(count + 1 <= MaxEntries);
    if (count) {
      unsigned pos = lowerBound(k);

      Key tmp_key = k;
      int cmp = 0;
      if (pos >= count || tmp_key.getLen() < prefix_key_.getLen()) {
        cmp = -1;
      } else {
        tmp_key.chunkBeginning(prefix_key_.getLen());
        cmp = tmp_key.compare(keys[pos], prefix_key_);
      }
      // only support one key one value, does not support one key multiple values
      // same value will overwrite the previous value with the same key
      if ((pos < count) && (cmp == 0)) {
        payloads[pos] = p;
        return;
      }
      memmove(keys + pos + 1, keys + pos, sizeof(Key) * (count - pos));
      memmove(payloads + pos + 1, payloads + pos, sizeof(Payload) * (count - pos));

      // get common prefix of key and other keys
      int new_prefix_len = k.commonPrefix(prefix_key_);
      k.chunkBeginning(new_prefix_len);
      keys[pos] = k;
      payloads[pos] = p;
      count++;
      // decide if we need to modify all the other keys
      if (new_prefix_len == prefix_key_.getLen()) {
        // insert directly
      } else {
        // modify all the keys, add the last several bytes of prefix to those keys
        assert(new_prefix_len < prefix_key_.getLen());
        uint16_t prefix_len = prefix_key_.getLen();
        for (int i = new_prefix_len; i < prefix_len; i++) {
          for (int j = 0; j < count; j++) {
            if (j == pos)
              continue;
            keys[j].addHead(prefix_key_);
          }
          prefix_key_.setLength(prefix_key_.getLen() - 1);
        }
      }
    } else {
      prefix_key_.setKeyStr(k.getKeyStr(), k.getLen());
      k.setKeyStr("", 0);
      keys[0] = k;
      payloads[0] = p;
      count++;
    }
//    for(int i = 0; i < count; i++) {
//      std::cout << "Partial Key: " << keys[i].getKeyStr() << " " << keys[i].getLen() << std::endl;
//    }
//    std::cout << "New Prefix: " << prefix_key_.getKeyStr() << " " << prefix_key_.getLen() << std::endl;
  }

  BTreeLeaf* split(Key& sep) {
    BTreeLeaf* newLeaf = new BTreeLeaf();
    newLeaf->count = count-(count/2);
    count = count-newLeaf->count;
    memcpy(newLeaf->keys, keys+count, sizeof(Key)*newLeaf->count);
    memcpy(newLeaf->payloads, payloads+count, sizeof(Payload)*newLeaf->count);
    // Set common prefix
    newLeaf->prefix_key_.setKeyStr(prefix_key_.getKeyStr(), prefix_key_.getLen());
    sep = keys[count-1];
    return newLeaf;
  }
};

struct BTreeInnerBase : public NodeBase {
  static const PageType typeMarker=PageType::BTreeInner;
};

struct BTreeInner : public BTreeInnerBase {

  Key prefix_key_;
  NodeBase* children[MaxEntries];
  Key keys[MaxEntries];

  BTreeInner() {
    count=0;
    type=typeMarker;
  }

  int64_t getSize() {
    int64_t key_size = prefix_key_.getSize();
    return key_size + sizeof(NodeBase *) * MaxEntries + key_size * MaxEntries;
  }

  bool isFull() { return count==(MaxEntries-1); };

  unsigned lowerBoundBF(Key k) {
    auto base=keys;
    unsigned n=count;
    while (n>1) {
      const unsigned half=n/2;
      int cmp = k.compare(base[half], prefix_key_);
      base = (cmp < 0) ? (base + half) : base;
      n -= half;
    }
    int cmp = k.compare(*base, prefix_key_);
    return static_cast<unsigned int>((cmp < 0)+ base - keys);
  }

  unsigned lowerBound(Key &k) {
    unsigned lower=0;
    unsigned upper=count;
    do {
      unsigned mid = ((upper - lower) / 2) + lower;
      Key &mid_key = keys[mid];
      int cmp = k.compare(mid_key, prefix_key_);
      if (cmp < 0) {
        upper=mid;
      } else if (cmp > 0) {
        lower = mid+1;
      } else {
        return mid;
      }
    } while (lower < upper);
    return lower;
  }

  BTreeInner* split(Key &sep) {
    BTreeInner* newInner=new BTreeInner();
    newInner->count=count-(count/2);
    count=count-newInner->count-1;
    sep=keys[count];
    memcpy(newInner->keys,keys+count+1,sizeof(Key)*(newInner->count+1));
    memcpy(newInner->children,children+count+1,sizeof(NodeBase*)*(newInner->count+1));
    newInner->prefix_key_ = prefix_key_;
    return newInner;
  }

  void insert(Key k,NodeBase* child) {
    assert(count <= MaxEntries - 1);
    unsigned pos=lowerBound(k);
    memmove(keys+pos+1,keys+pos,sizeof(Key)*(count-pos+1));
    memmove(children+pos+1,children+pos,sizeof(NodeBase*)*(count-pos+1));


    // get common prefix of key and other keys
    int new_prefix_len = k.commonPrefix(prefix_key_);
    k.chunkBeginning(new_prefix_len);
    keys[pos] = k;
    children[pos]=child;
    count++;
    // decide if we need to modify all the other keys
    if (new_prefix_len == prefix_key_.getLen()) {
      // insert directly
    } else {
      // modify all the keys, add the last several bytes of prefix to those keys
      assert(new_prefix_len < prefix_key_.getLen());
      uint16_t prefix_len = prefix_key_.getLen();
      prefix_key_.setLength(static_cast<uint16_t>(new_prefix_len));
      for (int i = new_prefix_len; i < prefix_len; i++) {
        for (int j = 0; j < count; j++) {
          if (j == pos)
            continue;
          keys[j].addHead(prefix_key_);
        }
        prefix_key_.removeHead();
      }
    }
    std::swap(children[pos],children[pos+1]);
  }
};


template<class Value>
struct BTree {
  std::atomic<NodeBase*> root;

  BTree() {
    root = new BTreeLeaf<Value>();
  }

  void makeRoot(Key k,NodeBase* leftChild,NodeBase* rightChild) {
    auto inner = new BTreeInner();
    inner->count = 1;
    inner->keys[0] = k;
    inner->children[0] = leftChild;
    inner->children[1] = rightChild;
    root = inner;
  }

  void yield(int count) {
    if (count>3)
      sched_yield();
    else
      _mm_pause();
  }

  void insert(Key k, Value v) {
    int restartCount = 0;
    restart:
    if (restartCount++)
      yield(restartCount);
    bool needRestart = false;

    // Current node
    NodeBase *node = root;
    uint64_t versionNode = node->readLockOrRestart(needRestart);
    if (needRestart || (node!=root)) goto restart;

    // Parent of current node
    BTreeInner *parent = nullptr;
    uint64_t versionParent;

    while (node->type==PageType::BTreeInner) {
      auto inner = static_cast<BTreeInner *>(node);

      // Split eagerly if full
      if (inner->isFull()) {
        // Lock
        if (parent) {
          parent->upgradeToWriteLockOrRestart(versionParent, needRestart);
          if (needRestart) goto restart;
        }
        node->upgradeToWriteLockOrRestart(versionNode, needRestart);
        if (needRestart) {
          if (parent)
            parent->writeUnlock();
          goto restart;
        }
        if (!parent && (node != root)) { // there's a new parent
          node->writeUnlock();
          goto restart;
        }
        // Split
        Key sep; BTreeInner *newInner = inner->split(sep);
        if (parent)
          parent->insert(sep,newInner);
        else
          makeRoot(sep,inner,newInner);
        // Unlock and restart
        node->writeUnlock();
        if (parent)
          parent->writeUnlock();
        goto restart;
      }

      if (parent) {
        parent->readUnlockOrRestart(versionParent, needRestart);
        if (needRestart) goto restart;
      }

      parent = inner;
      versionParent = versionNode;
      int c = inner->lowerBound(k);

      node = inner->children[inner->lowerBound(k)];
      inner->checkOrRestart(versionNode, needRestart);
      if (needRestart) goto restart;
      versionNode = node->readLockOrRestart(needRestart);
      if (needRestart) goto restart;
    }

    auto leaf = static_cast<BTreeLeaf<Value>*>(node);

    // Split leaf if full
    if (leaf->count == MaxEntries) {
      // Lock
      if (parent) {
        parent->upgradeToWriteLockOrRestart(versionParent, needRestart);
        if (needRestart) goto restart;
      }
      node->upgradeToWriteLockOrRestart(versionNode, needRestart);
      if (needRestart) {
        if (parent) parent->writeUnlock();
        goto restart;
      }
      if (!parent && (node != root)) { // there's a new parent
        node->writeUnlock();
        goto restart;
      }
      // Split
      Key sep; BTreeLeaf<Value>* newLeaf = leaf->split(sep);
      if (parent)
        parent->insert(sep, newLeaf);
      else
        makeRoot(sep, leaf, newLeaf);
      // Unlock and restart
      node->writeUnlock();
      if (parent)
        parent->writeUnlock();
      goto restart;
    } else {
      // only lock leaf node
      node->upgradeToWriteLockOrRestart(versionNode, needRestart);
      if (needRestart) goto restart;
      if (parent) {
        parent->readUnlockOrRestart(versionParent, needRestart);
        if (needRestart) {
          node->writeUnlock();
          goto restart;
        }
      }
      leaf->insert(k, v);
      node->writeUnlock();
      return; // success
    }
  }

  bool lookup(Key k, Value& result) {
    int restartCount = 0;
    restart:
    if (restartCount++)
      yield(restartCount);
    bool needRestart = false;

    NodeBase* node = root;
    uint64_t versionNode = node->readLockOrRestart(needRestart);
    if (needRestart || (node!=root)) goto restart;

    // Parent of current node
    BTreeInner *parent = nullptr;
    uint64_t versionParent;

    while (node->type==PageType::BTreeInner) {
      auto inner = static_cast<BTreeInner *>(node);

      if (parent) {
        parent->readUnlockOrRestart(versionParent, needRestart);
        if (needRestart) goto restart;
      }

      parent = inner;
      versionParent = versionNode;

      node = inner->children[inner->lowerBound(k)];
      inner->checkOrRestart(versionNode, needRestart);
      if (needRestart) goto restart;
      versionNode = node->readLockOrRestart(needRestart);
      if (needRestart) goto restart;
    }

    BTreeLeaf<Value>* leaf = static_cast<BTreeLeaf<Value>*>(node);
    unsigned pos = leaf->lowerBound(k);
    bool success = false;
    int cmp = k.compare(leaf->keys[pos], leaf->prefix_key_);
    if ((pos<leaf->count) && (cmp == 0)) {
      success = true;
      result = leaf->payloads[pos];
    }
    if (parent) {
      parent->readUnlockOrRestart(versionParent, needRestart);
      if (needRestart) goto restart;
    }
    node->readUnlockOrRestart(versionNode, needRestart);
    if (needRestart) goto restart;

    return success;
  }

  uint64_t scan(Key k, int range, Value* output) {
    int restartCount = 0;
    restart:
    if (restartCount++)
      yield(restartCount);
    bool needRestart = false;

    NodeBase* node = root;
    uint64_t versionNode = node->readLockOrRestart(needRestart);
    if (needRestart || (node!=root)) goto restart;

    // Parent of current node
    BTreeInner *parent = nullptr;
    uint64_t versionParent;

    while (node->type==PageType::BTreeInner) {
      auto inner = static_cast<BTreeInner*>(node);

      if (parent) {
        parent->readUnlockOrRestart(versionParent, needRestart);
        if (needRestart) goto restart;
      }

      parent = inner;
      versionParent = versionNode;

      node = inner->children[inner->lowerBound(k)];
      inner->checkOrRestart(versionNode, needRestart);
      if (needRestart) goto restart;
      versionNode = node->readLockOrRestart(needRestart);
      if (needRestart) goto restart;
    }

    BTreeLeaf<Value>* leaf = static_cast<BTreeLeaf<Value>*>(node);
    unsigned pos = leaf->lowerBound(k);
    int count = 0;
    for (unsigned i=pos; i<leaf->count; i++) {
      if (count==range)
        break;
      output[count++] = leaf->payloads[i];
    }

    if (parent) {
      parent->readUnlockOrRestart(versionParent, needRestart);
      if (needRestart) goto restart;
    }
    node->readUnlockOrRestart(versionNode, needRestart);
    if (needRestart) goto restart;

    return count;
  }

  int64_t getSize() {
    int64_t size = 0;
    NodeBase *node = root;
    std::vector<NodeBase *> l;
    l.push_back(root);
    int node_cnt = 1;

    while (!l.empty()) {
      NodeBase *top = l.front();
      if (top->type == PageType::BTreeInner) {
        auto node = reinterpret_cast<BTreeInner *>(top);
        size += node->getSize();
        for (int i = 0; i < node->count; i++) {
          l.push_back(node->children[i]);
          node_cnt++;
        }
      } else {
        auto node = reinterpret_cast<BTreeLeaf<Value> *>(top);
        size += node->getSize();
      }
      l.erase(l.begin());
    }
    std::cout << "CPS Btree Node Num=" << node_cnt << std::endl;
    return size;
  }

};

}