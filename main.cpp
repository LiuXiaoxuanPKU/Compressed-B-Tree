#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include "include/CPSBTreeOLC.h"
#include "include/BTreeOLC.h"
#include "include/tlx/btree_map.hpp"

static const std::string email_dir = "/Users/xiaoxuanliu/Documents/Research/OPE_dataset/emails.txt";

double getNow() {
  struct timeval tv;
  gettimeofday(&tv, 0);
  return tv.tv_sec + tv.tv_usec / 1000000.0;
}

int64_t loadKeys(const std::string& file_name,
                 std::vector<std::string> &keys,
                 std::vector<std::string> &keys_shuffle,
                 int64_t &total_len) {
  std::ifstream infile(file_name);
  std::string key;
  total_len = 0;
  int cnt = 0;
  while (infile.good() && cnt < 1000000) {
    infile >> key;
    cnt++;
    keys.push_back(key);
    keys_shuffle.push_back(key);
    total_len += key.length();
  }
  std::random_shuffle(keys_shuffle.begin(), keys_shuffle.end());
  return total_len;
}

int main() {
  std::vector<std::string> emails;
  std::vector<std::string> emails_shuffle;
  int64_t total_key_len = 0;
  loadKeys(email_dir, emails, emails_shuffle, total_key_len);
  auto cpstree = new cpsbtreeolc::BTree<std::string>();
  auto btree = new btreeolc::BTree<std::string, std::string>();
  typedef tlx::btree_map<std::string, std::string, std::less<std::string> > btree_type;
  btree_type* tlx_btree = new btree_type();

  int insert_cnt = 0;
  double cps_insert_start = getNow();
  for (const auto &email : emails) {
    insert_cnt++;
    cpsbtreeolc::Key key;
    key.setKeyStr(email.c_str(), email.length());
    cpstree->insert(key, email);
  }
  double cps_insert_end = getNow();
  double cps_tput = insert_cnt / (cps_insert_end - cps_insert_start) / 1000000; // M items / s

  insert_cnt = 0;
  double btree_insert_start = getNow();
  for (const auto &email : emails) {
    insert_cnt++;
    btree->insert(email, email);
  }
  double btree_insert_end = getNow();
  double btree_tput = insert_cnt / (btree_insert_end - btree_insert_start) / 1000000; // M items / s

  insert_cnt = 0;
  double tlxbtree_insert_start = getNow();
  for (const auto &email : emails) {
    insert_cnt++;
    tlx_btree->insert2(email, email);
  }
  double tlxbtree_insert_end = getNow();
  double tlxbtree_tput = insert_cnt / (tlxbtree_insert_end - tlxbtree_insert_start) / 1000000; // M items / s

  int64_t cpstree_size = cpstree->getSize();
  int64_t btree_size = btree->getSize() + total_key_len;
  int64_t tlxbtree_size = 256 * tlx_btree->get_stats().nodes() + total_key_len;
  std::cout << total_key_len << "\nCompressed BTree = " << cpstree_size << std::endl
                          << "Compressed BTree Throughput = " << cps_tput << " M items/s " << std::endl
                          << "Victor BTree=" << btree_size << std::endl
                          << "Victor BTree Throughput = " << btree_tput << " M items/s " << std::endl
                          << "TLXBTree=" << tlxbtree_size << std::endl
                          << "TLXBTree Throughput = " << tlxbtree_tput << " M items/s " << std::endl;

  int cnt = 0;
  for (const auto &email : emails_shuffle) {
    cnt += 1;
//    if (cnt % 10000 == 0) std::cout << cnt << std::endl;
    std::string cpsvalue;
    std::string btreevalue;
    cpsbtreeolc::Key key;
    key.setKeyStr(email.c_str(), email.length());
    cpstree->lookup(key, cpsvalue);
    btree->lookup(email, btreevalue);
    assert(cpsvalue.compare(email) == 0);
    assert(btreevalue.compare(email) == 0);
  }
}