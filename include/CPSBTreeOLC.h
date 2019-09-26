#pragma once

#include <cassert>
#include <cstring>
#include <atomic>
#include <immintrin.h>
#include <sched.h>
#include <algorithm>

namespace cpsbtreeolc {

enum class PageType : uint8_t { BTreeInner=1, BTreeLeaf=2 };

static const uint16_t MAX_KEY_SIZE = 8;
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
 public:
  // if prefix is too long
  // key stores the pointer to the prefix
  char key[MAX_KEY_SIZE + 1];
  uint16_t part_len_;
  char *overflow_key;

  Key() {
    part_len_ = 0;
    overflow_key = nullptr;
  }
  ~Key() {
    if (part_len_ > MAX_KEY_SIZE  && overflow_key != nullptr) {
      //delete []overflow_key;
      //overflow_key = nullptr;
    }
  }

  Key(Key &right) {
    part_len_ = 0;
    overflow_key = nullptr;

    setKeyStr(right.getKeyStr(), right.part_len_);
  }

  int64_t getSize() { // in bytes
    int64_t re = MAX_KEY_SIZE + 1 + sizeof(uint16_t) + sizeof(char *);
    if (overflow_key != nullptr)
      re += part_len_;
    return re;
  }

  void setLength(uint16_t new_len) {
    if (part_len_ > MAX_KEY_SIZE && new_len <= MAX_KEY_SIZE) {
      memcpy(key, overflow_key, new_len);
      delete []overflow_key;
      overflow_key = nullptr;
    }
    part_len_ = new_len;
  }

  void setKeyStr(const char *str, uint16_t len) {
    if (overflow_key != nullptr) {
      delete []overflow_key;
      overflow_key = nullptr;
    }

    if (len > MAX_KEY_SIZE) {
      overflow_key = new char[len + 1];
      memcpy(overflow_key, str, len);
      overflow_key[len] = '\0';
    } else {
      for (int i = 0; i < len; i++) {
        key[i] = str[i];
      }
      key[len] = '\0';
    }
    part_len_ = len;
  }

  const char *getKeyStr() const {
    if (part_len_ > MAX_KEY_SIZE) {
      return overflow_key;
    }
    return key;
  }

  int compare(Key &right, Key &prefix) {
    const char *my_str = getKeyStr();
    const char *prefix_str = prefix.getKeyStr();
    const char *right_str = right.getKeyStr();

    // Concatenate to get the full key
    char *full_key = new char[prefix.part_len_ + right.part_len_ + 1];
    memcpy(full_key, prefix_str, prefix.part_len_);
    memcpy(full_key + prefix.part_len_, right_str, right.part_len_);
    int full_key_len = prefix.part_len_ + right.part_len_;
    full_key[full_key_len] = '\0';
    int cmp = strcmp(my_str, full_key);
    delete []full_key;
    return cmp;
  }

  int commonPrefix(Key &right) {
    uint16_t len = std::min(part_len_, right.part_len_);
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
    char new_head_char = prefix.getKeyStr()[prefix.part_len_ - 1];
    // already overflow
    if (part_len_ > MAX_KEY_SIZE) {
      memmove(overflow_key + 1, overflow_key, part_len_);
      overflow_key[0] = new_head_char;
    } else if (part_len_ == MAX_KEY_SIZE) {
      // overflow
      overflow_key = new char[MAX_KEY_SIZE + 2];
      overflow_key[0] = new_head_char;
      memcpy(overflow_key + 1, key, MAX_KEY_SIZE);
      overflow_key[MAX_KEY_SIZE + 1] = '\0';
    } else {
      // no overflow
      memmove(key + 1, key, part_len_);
      key[0] = new_head_char;
    }
    part_len_ += 1;
  }

  void removeHead() {
    if (part_len_ > MAX_KEY_SIZE + 1) {
      memmove(overflow_key, overflow_key + 1, part_len_ - 1);
      overflow_key[part_len_ - 1] = '\0';
    } else if (part_len_ == MAX_KEY_SIZE + 1) {
      memcpy(key, overflow_key + 1, MAX_KEY_SIZE);
      delete []overflow_key;
      overflow_key = nullptr;
      overflow_key[MAX_KEY_SIZE] = '\0';
    } else {
      memmove(key, key + 1, part_len_ - 1);
      key[part_len_ - 1] = '\0';
    }
    if (part_len_ > 0) part_len_ -= 1;
  }

  // remove the head number of bytes
  void chunkBeginning(uint16_t cnt) {
    assert(cnt <= part_len_);
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
      Key mid_key;
      mid_key = keys[mid];

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

  void insert(Key &k,Payload p) {
    assert(count + 1 <= MaxEntries);
    if (count) {
      unsigned pos = lowerBound(k);

      Key tmp_key = k;
      int cmp = 0;
      if (pos >= count || tmp_key.part_len_ < prefix_key_.part_len_) {
        cmp = -1;
      } else {
        tmp_key.chunkBeginning(prefix_key_.part_len_);
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
      if (new_prefix_len == prefix_key_.part_len_) {
        // insert directly
      } else {
        // modify all the keys, add the last several bytes of prefix to those keys
        assert(new_prefix_len < prefix_key_.part_len_);
        uint16_t prefix_len = prefix_key_.part_len_;
        for (int i = new_prefix_len; i < prefix_len; i++) {
          for (int j = 0; j < count; j++) {
            if (j == pos)
              continue;
            keys[j].addHead(prefix_key_);
          }
          prefix_key_.setLength(prefix_key_.part_len_ - 1);
        }
      }
    } else {
      prefix_key_.setKeyStr(k.getKeyStr(), k.part_len_);
      k.setKeyStr("", 0);
      keys[0] = k;
      payloads[0] = p;
      count++;
    }
  }

  BTreeLeaf* split(Key& sep) {
    BTreeLeaf* newLeaf = new BTreeLeaf();
    newLeaf->count = count-(count/2);
    count = count-newLeaf->count;
    memcpy(newLeaf->keys, keys+count, sizeof(Key)*newLeaf->count);
    memcpy(newLeaf->payloads, payloads+count, sizeof(Payload)*newLeaf->count);
    // Set common prefix
    newLeaf->prefix_key_.setKeyStr(prefix_key_.getKeyStr(), prefix_key_.part_len_);
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
    if (new_prefix_len == prefix_key_.part_len_) {
      // insert directly
    } else {
      // modify all the keys, add the last serveral bytes of prefix to those keys
      assert(new_prefix_len < prefix_key_.part_len_);
      prefix_key_.setLength(static_cast<uint16_t>(new_prefix_len));
      uint16_t prefix_len = prefix_key_.part_len_;
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
    //count++;
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

  void insert(Key &k, Value v) {
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

  bool lookup(Key &k, Value& result) {
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