/************************************************************************************
  C++ code for one-time preprocessing of the input text documents
     removes words containing numbers and/or special characters
     converts left over words into lower case
     removes stop words
     removes less frequent words (total frequency across all documents < 10)
     creates a vocabulary which is a map from a word to a unique number
     outputs are printed in the following format:
          M (number of documents)
          V (number of words in vocabulary)
          N[M] (M numbers representing number of words in each document)
          M lines (each containing a list of word numbers for that document)
   Outputs are redirected to a file which serves as input to the LDA sampling code
*************************************************************************************/

#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>
#include <algorithm>
#include <map>
#include <vector>
#include <regex>
#include <set>
#include <numeric>

#define THRESHOLD 10

// to check whether a given character is a digit or not
bool isNum(char c) {
  if (std::isdigit(c)) {
    return true;
  }
  return false;
}

// to populate the stop words vector by reading from a file
void populateStopWords(std::set<std::string>& stopWords) {
  std::ifstream file("./stopWords.txt");
  std::string word;
  while (std::getline(file, word)) {
    stopWords.insert(word);
  }
  file.close();
}

// to poulate the file names of the TOI data set
void populateFileNames(std::vector<std::string>& filenames) {
  std::ifstream file("./filenames.txt");
  std::string fn;
  while (std::getline(file, fn)) {
    filenames.push_back(fn);
  }
  file.close();
}

int main() {
  std::string word;
  std::map<std::string, int> vocabulary;
  std::regex reg("\\s+");
  std::set<std::string> stopWords;
  std::vector<std::string> filenames;
  std::vector<std::vector<std::string> > docWords;
  int i, j;

  // the directory which stores all the text files of the TOI dataset
  std::string datadir = "./data/";

  // populating the vector of stop words
  populateStopWords(stopWords);

  // populating the files names from the TOI dataset
  populateFileNames(filenames);

  // number of documents which we will use
  int M = 40000;
  std::cout << M << std::endl;

  // number of words in each document
  std::vector<int> N(M);

  // processing each of the M text files
  for (i = 0; i < M; i++) {
    // getting the complete path to the file
    std::string filename = datadir + filenames[i];

    // obatining a file handler to read the file
    std::ifstream file(filename);

    // to store the words in the current file
    std::vector<std::string> currWords;

    // reading from the file, one word at a time
    while (file >> word) {
      // discarding the characters of the word which are digits
      word.erase(std::remove_if(word.begin(), word.end(), &isNum), word.end());

      // if the word still has some characters left
      if (word.length() > 0) {
        // converting to lower case, replacing non-alphabet characters with whitespace
        std::for_each(word.begin(), word.end(), [](char& c) {
          if (std::isalpha(c)) {
            c = std::tolower(c);
          }
          else {
            c = ' ';
          }
        });

        // splitting the word by whitespace
        std::sregex_token_iterator iter(word.begin(), word.end(), reg, -1);
        std::sregex_token_iterator end;

        // final list of tokenised lower-case words
        std::vector<std::string> words(iter, end);

        // iterating through this list of words
        for (auto a : words) {
          // if this is not an empty word (might happen because of tokenising)
          if (a.length() > 0) {
            // if it is not a stop word
            if (stopWords.find(word) == stopWords.end()) {
              // add this word to the vocabulary and increment its frequency/count
              vocabulary[a]++;

              // add this word to the current document's list of words
              currWords.push_back(a);
            }
          }
        }
      }
    }

    // append the list of words for this document to the vector of words of all documents
    docWords.push_back(currWords);

    // getting the number of words in this document
    N[i] = currWords.size();
  }

  // updating the vocabulary by removing less-frequent words (words occurring at a frequency less than THRESHOLD = 10)
  std::map<std::string, int>::iterator temp_it;
  for (auto it = vocabulary.begin(); it != vocabulary.end(); it++) {
    temp_it = it;
    if (temp_it->second < THRESHOLD) {
      vocabulary.erase(temp_it);
    }
  }

  // number of words in the final vocabulary
  int V = vocabulary.size();
  std::cout << V << std::endl;

  // assigning a unique integer id to each word in the vocabulary
  i = 0;
  for (auto& word : vocabulary) {
    word.second = i;
    i++;
  }

  // updating the word lists of the individual documents
  for (i = 0; i < M; i++) {
    // initial number of words present in this document
    int tempnum = N[i];
    for (auto j = docWords[i].begin(); j != docWords[i].end(); j++) {
      // if the word in not present in the vicabulary
      if (vocabulary.find(*j) == vocabulary.end()) {
        // remove it from the current document's list of words
        docWords[i].erase(j);

        // update the iterator
        j--;

        // update the count of number of words left in the document
        tempnum--;
      }
    }

    // storing the updated count of number of words in this document
    N[i] = tempnum;
  }

  // number of words in each document
  for (i = 0; i < M; i++) {
    std::cout << N[i] << " ";
  }
  std::cout << std::endl;

  // list of ids of the words in each document
  for (i = 0; i < M; i++) {
    for (j = 0; j < N[i]; j++) {
      std::cout << vocabulary[docWords[i][j]] << " ";
    }
    std::cout << std::endl;
  }

  // generating a summary of the processed dataset
  int initial_sum = 0;
  int totWords = accumulate(N.begin(), N.end(), initial_sum);

  std::ofstream sum_file("./input_data_summary.txt");
  sum_file << M << " documents\n";
  sum_file << V << " words in vocabulary\n";
  sum_file << totWords << " number of words in entire input\n";
  sum_file << *std::max_element(N.begin(), N.end()) << " max words in a document\n";
  sum_file << *std::min_element(N.begin(), N.end()) << " min words in a document\n";
  sum_file << ((double) totWords)/M << " average words in a document\n";
  sum_file.close();

  return 0;
}
