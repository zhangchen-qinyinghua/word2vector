//  Copyright 2013 Google Inc. All Rights Reserved.
//
//  Licensed under the Apache License, Version 2.0 (the "License");
//  you may not use this file except in compliance with the License.
//  You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
//  Unless required by applicable law or agreed to in writing, software
//  distributed under the License is distributed on an "AS IS" BASIS,
//  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
//  See the License for the specific language governing permissions and
//  limitations under the License.
//
//
//  本注释，全局变量的注释在定义处
//
//  阅读W2V，从main函数开始，全流程大概是这样的
//  1.参数初始化
//  2.单词表初始化
//  3.并行训练
//  4.(可选)k-means聚类
//  5.输出
//
//  没有注释k-means部分
//  没有注释skip-gram部分，这一部分跟cbow是一一对应的

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>

#define MAX_STRING 100 //本代码中出现的一切c字符串的最大长度
#define EXP_TABLE_SIZE 1000 // EXP_TABLE是sigmoid函数表格，这个函数并不是网络激活函数，其作用在后文中体现
#define MAX_EXP 6 //sigmoid相关函数表格只计算-6~+6之间的值，这个范围之外认为sigmoid函数的值是0
#define MAX_SENTENCE_LENGTH 1000 //一个句子一直没有遇到换行符，就在1000个单词处作为为句子末端
#define MAX_CODE_LENGTH 40 //这里是code和point的统一最大长度，vocab_word里面的code和point组使用了定长
//定长为40满足节点数2^41

const int vocab_hash_size = 30000000;  // Maximum 30 * 0.7 = 21M words in the vocabulary
//与词汇表配套的hash表最大容量

typedef float real;                    // Precision of float numbers
//全文使用real作为实数类型

//一个单词及其相关信息的结构体
struct vocab_word {
	long long cn;//the word frequency
	//cn是单词频数
	int *point;
	//point表示这个词汇对应的辅助向量列的index序列，其长度也是codelen
	//辅助向量全部定义在syn1中
	char *word, *code, codelen;
	//字段word就是词汇本身
	//code是这个词汇对应的huffman编码，由0和1组成
	//codelen表示code的长度，huffman是不定长的编码
	//上述定义参见hierarchicql softmax原理
};

char train_file[MAX_STRING], output_file[MAX_STRING];
//训练语料文件名和输出文件名，必须指定
char save_vocab_file[MAX_STRING], read_vocab_file[MAX_STRING];
//词汇表输出文件和读取词汇表的文件
//都是可选的
struct vocab_word *vocab;
//动态变化的vocab_word词汇表
int binary = 0, cbow = 0, debug_mode = 2, window = 5, min_count = 5, num_threads = 1, min_reduce = 1;
//binary，输出是否二进制
//cbow，是否使用连续词袋模型，否则使用skip-gram模型
//window，所谓的窗口，与一个单词上下文的取得有关，由用户指定，默认为5，具体见训练部分代码TrainModelThread
//min_count，舍弃低频词的阈值
//min_reduce，一种低频词舍弃方法，参见ReduceVocab方法
int *vocab_hash;
//与词汇表配套的hash表，用于根据word快速获得其在词汇表中的位置，从而的到word对应的定义于vocab_word结构里面的信息
long long vocab_max_size = 1000, vocab_size = 0, layer1_size = 100;
//vocab_max_size是动态词汇表的最大容量
//vocab_size是动提案词汇表的实际容量
//layer1_size是每一个word产生vector的维数，根据用户跟定参数定义。看他的命名，认为这是第一层的节点数量
//因为第一层是输入word的vector之和(对应cbow模型)或者某个word对应的vector(对应n-gram模型)
long long train_words = 0, word_count_actual = 0, file_size = 0, classes = 0;
//train_words是所有有效词汇的频率之和，有效词汇的意义在于，对于词频小的词做了舍弃
//file_size是训练语料文件的大小，这个值用于并行化时负载平衡
real alpha = 0.025, starting_alpha, sample = 0;
//alpha学习率，starting_alpha初始学习率,可指定
//sample高频词亚采样所用的参数，见词汇表初始化部分
real *syn0, *syn1, *syn1neg, *expTable;
// 上述三个变量
// syn0就是W2V的最终结果，大小是vector维数*输出词汇数目
// syn1就是全部辅助变量，个数比syn0少了一个，也就是大小为vector维数*(输出词汇数目-1)
// syn1neg是随机负采样得到的向量表
// expTable是预先计算的sigmoid函数值，这个值的定义具体见main函数的底部
// syn0 is the word_vector NET
// syn0 get vocab_size vectors, each vector is in layer1_size dimentions
// syn1 is a assist set of vectors
clock_t start;
//start用于计时，仅仅是看看程序运行速度

int hs = 1, negative = 0;
//参数hs表示是否使用hierarchical softmax，就是huffman树和逻辑回归结合的策略，具体见w2v原理
//negartive表示是否使用随机负采样，如果这个值大于0则表示每一个word和上下文要负采样的个数
const int table_size = 1e8;
int *table;
//table_size和table对应随机负采样算法所用的配套工具
//table是一个映射，把0~1e8-1的值，映射为0~vocab_size-1的值
//table用于随机负采样，对每一个训练样本，取一个负样本
//具体使用table的方法是，对每一个训练样本随机产生一个0~1e8-1之间的数字
//利用table把他映射成某个词的index
//这个词就是负样本，具体负样本的使用和生成跟模型有关，请看训练部分代码或者算法原理论文
//具体见下面的InitUnigramTable函数定义

//初始化负采样辅助table
//保证一个单词被选为负样本的概率为
//(单词频数)^(power)/(所有的单词，(单词频数)^(power)之和,也就是train_word_pow的值)
//这里power=0.75
void InitUnigramTable() {
	int a, i;
	long long train_words_pow = 0;
	real d1, power = 0.75;
	//power=0.75
	//目前来看这是一个magic number
	//不明这个值得选取原理
	//这个值得意义是
	//一个单词被选为负样本的概率为
	//(单词频数)^(power)/(所有的单词，(单词频数)^(power)之和,也就是train_word_pow的值)
	table = (int *)malloc(table_size * sizeof(int));
	for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
	//对所有的单词，(单词频数)^(power)之和,也就是train_word_pow的值
	i = 0;
	d1 = pow(vocab[i].cn, power) / (real)train_words_pow;
	//实际为选中词汇i为负样本的概率
	//累加这个值是为了保证pow(vocab[i].cn, power) / (real)train_words_pow的table值为词汇i的indx
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / (real)table_size > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / (real)train_words_pow;
		}
		if (i >= vocab_size) i = vocab_size - 1;
	}
}

// Reads a single word from a file, assuming space + tab + EOL to be word boundaries
// 如上原注释所言
// 读取单个词汇
// 词汇边界有3，空格(' ')，制表符('\t')，和EOL(EOF、'\n')
// 需要注意这里面对换行符的处理
// 一是不认为\r为有效符号
// 二是单个换行符号\n被认为是有效符号，并且记为词汇"</s>"
// 三是遇到换行符作为结尾，换行符将被回退到输入流，形成一个单个换行符，他在下一次读入的时候成为"</s>"
// 换行符被替换为"</s>"是为了标记一行的结束
// 由于w2v的训练是以行为单位进行的，也就是根据</s>的分割进行
// 所以一个句子一行的训练语料应该是合适的
void ReadWord(char *word, FILE *fin) {
	int a = 0, ch;
	//while中的判断含义如原注释，文件结尾作为词汇边界
	while (!feof(fin)) {
		ch = fgetc(fin);
		//char(13)是回车，即\r
		//回车与换行当然不一样，其发明有历史原因，在不同的操作系统有不同的使用约定
		//对回车符号，仅仅简单的略过，因为一般情况下，\r不单独表示换行，如果要换行往往会有\n
		//比如win系统的换行是\r\n
		//unix则是\n
		//但是mac下的换行为\r
		//因此，这里要求我们对输入文件做换行符号转换，确保以\n换行
		if (ch == 13) continue;
		//词汇边界检测
		if ((ch == ' ') || (ch == '\t') || (ch == '\n')) {
			if (a > 0) {
				if (ch == '\n') ungetc(ch, fin);
				//注意换行符被回退到输入流里面
				break;
			}
			//单个换行符号检测
			//如果不是单个换行符号，对其他类型的边界做略过
			if (ch == '\n') {
				strcpy(word, (char *)"</s>");
				return;
			} else continue;
		}
		word[a] = ch;
		a++;
		//这里，如果单个词汇的长度过长，超过MAX_STRING
		//将对词汇做截断处理
		//不过仍然要把输入流读完，直到单词边界为止
		if (a >= MAX_STRING - 1) a--;   // Truncate too long words
	}
	//根据c字符串的约定，结尾置0
	word[a] = 0;
}

// Returns hash value of a word
// 计算word的hash函数，直接用这个hash值在hash表里面定位
// 定位冲突使用线性探测
// 计算公式请看代码
int GetWordHash(char *word) {
	unsigned long long a, hash = 0;
	for (a = 0; a < strlen(word); a++) hash = hash * 257 + word[a];
	hash = hash % vocab_hash_size;
	return hash;
}

// Returns position of a word in the vocabulary; if the word is not found, returns -1
// 根据hash表快速的到词汇word在词汇表里面的位置，从而通过word查到vocab_word里面定义的词汇相关信息
// word没找到返回-1
int SearchVocab(char *word) {
	//获取hash值
	unsigned int hash = GetWordHash(word);
	while (1) {
		//线性探测词汇的位置
		//就是从初始位置开始一个一个向下比较直到遇到word为止
		//或者遇到-1，表示当前word并不在表里面
		if (vocab_hash[hash] == -1) return -1;
		if (!strcmp(word, vocab[vocab_hash[hash]].word)) return vocab_hash[hash];
		hash = (hash + 1) % vocab_hash_size;
	}
	//经过线性探测，没有找到word
	return -1;
}

// Reads a word and returns its index in the vocabulary
// 从fin中读取一个单词，同时查找返回他在词汇表里的位置
int ReadWordIndex(FILE *fin) {
	char word[MAX_STRING];
	ReadWord(word, fin);
	if (feof(fin)) return -1;
	return SearchVocab(word);
}

// Adds a word to the vocabulary
// 将词汇word放入vocab_word结构加入到词汇表里面
// 同时将词汇的位置记录在hash表里面方便定位
int AddWordToVocab(char *word) {
	unsigned int hash, length = strlen(word) + 1;
	//首先，截断单词，不能使之超过MAX_STRING的长度
	if (length > MAX_STRING) length = MAX_STRING;
	//vocab_word结构的定义见代码开头
	//vocab是vocab_word的动态增长的线性表，最开始有1000个
	//直接将单词插入其中
	vocab[vocab_size].word = (char *)calloc(length, sizeof(char));
	strcpy(vocab[vocab_size].word, word);
	vocab[vocab_size].cn = 0;
	vocab_size++;
	// Reallocate memory if needed
	// 单词数量超出则重新分配内存
	if (vocab_size + 2 >= vocab_max_size) {
		vocab_max_size += 1000;
		vocab = (struct vocab_word *)realloc(vocab, vocab_max_size * sizeof(struct vocab_word));
	}
	//获取单词的hash
	//hash表的长度是一定的
	//hash值经过了取模运算，直接用于定位
	hash = GetWordHash(word);
	//线性探测的方法解决hash表的冲突
	//也就是从hash的位置开始，一个一个向下检测，遇到没有冲突(值-1)的位置就插入当前单词
	//插入的是当前单词在词汇表里面的index位置
	while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
	vocab_hash[hash] = vocab_size - 1;
	//返回当前词汇在词汇表里面的位置
	return vocab_size - 1;
}

// Used later for sorting by word counts
// 词汇比较函数
// 注意这里大小顺序，是用b减去a
// 使用这个函数比较词汇，词频大的排在前面
int VocabCompare(const void *a, const void *b) {
	return ((struct vocab_word *)b)->cn - ((struct vocab_word *)a)->cn;
}

// Sorts the vocabulary by frequency using word counts
// 对词汇表重排
// 做了这么几件事情
// 1.按照频数从大到小排列词汇表
// 2.丢弃掉频数小于min_count的词汇
// 3.重新构建hash表
// 4.计算出train_words的值，这个值是所有词汇表里面有效的词的频数之和
// 5.为每个词汇的huffman编码初始化
void SortVocab() {
	int a, size;
	unsigned int hash;
	// Sort the vocabulary and keep </s> at the first position
	qsort(&vocab[1], vocab_size - 1, sizeof(struct vocab_word), VocabCompare);
	// 按照频数从大到小排列词汇表
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	//清理hash表
	size = vocab_size;
	train_words = 0;
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		// 舍弃频数小的词汇
		if (vocab[a].cn < min_count) {
			vocab_size--;
			free(vocab[vocab_size].word);
		} else {
			// Hash will be re-computed, as after the sorting it is not actual
			// 获取hash值作为初始位置
			hash=GetWordHash(vocab[a].word);
			// 线性探测解决冲突
			while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
			// hash表存放词汇位置信息
			vocab_hash[hash] = a;
			// train_words是所有词汇表里面有效的词的频数之和
			train_words += vocab[a].cn;
		}
	}
	//精简词汇表的大小
	vocab = (struct vocab_word *)realloc(vocab, (vocab_size + 1) * sizeof(struct vocab_word));
	// Allocate memory for the binary tree construction
	for (a = 0; a < vocab_size; a++) {
		vocab[a].code = (char *)calloc(MAX_CODE_LENGTH, sizeof(char));
		vocab[a].point = (int *)calloc(MAX_CODE_LENGTH, sizeof(int));
	}
}

// Reduces the vocabulary by removing infrequent tokens
// 舍弃掉词频小于等于min_reduce的单词
// 重构hash表
// min_reduce的初始值是1
// 每次执行本函数，min_reduce的值增加1
void ReduceVocab() {
	int a, b = 0;
	unsigned int hash;
	for (a = 0; a < vocab_size; a++) if (vocab[a].cn > min_reduce) {
		//单词a的词频大于min_reduce，不舍弃
		//插入到词汇表位置b
		vocab[b].cn = vocab[a].cn;
		vocab[b].word = vocab[a].word;
		b++;
		//否则，舍弃单词a
	} else free(vocab[a].word);
	vocab_size = b;
	//重构hash表
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	for (a = 0; a < vocab_size; a++) {
		// Hash will be re-computed, as it is not actual
		hash = GetWordHash(vocab[a].word);
		while (vocab_hash[hash] != -1) hash = (hash + 1) % vocab_hash_size;
		vocab_hash[hash] = a;
	}
	fflush(stdout);
	// 每次执行本函数，min_reduce的值增加1
	min_reduce++;
}

// Create binary Huffman tree using the word counts
// Frequent words will have short uniqe binary codes
// 霍夫曼树构建，vocab_word结构中编码部分的填充
// 关于这一部分的理解难以尽述，请看原理或者作者论文
// huffman树的构建使用贪心法，构建过程见函数定义，具体证明请自查
void CreateBinaryTree() {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	//
	//min1i和min2i表示两个count最小的树的index
	char code[MAX_CODE_LENGTH];
	long long *count = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	//count是所有原始词代表的树的权，和每次组合后代表的权，请先了解huffman的构建法再理解本段代码
	//预计count大小是2*vocab_size-1
	//这里用了一个更大的数组
	long long *binary = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	//binary用于记录所有节点对应的huffman编码，其值为0或者1
	//所以这里binary用longlong是很奇怪的，用char就够了
	//难道是为了对齐？
	long long *parent_node = (long long *)calloc(vocab_size * 2 + 1, sizeof(long long));
	//parent_node用于记录所有子树的父亲节点
	//首先对最终将成为叶子节点的词汇树赋权，权值就是词频
	for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
	//剩下节点赋权一个大数，这是为了获取最小的两个权值方便
	for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
	pos1 = vocab_size - 1;
	pos2 = vocab_size;
	// Following algorithm constructs the Huffman tree by adding one node at a time
	// 每次两棵树合并，合并voca_size-1次刚好构建完成
	for (a = 0; a < vocab_size - 1; a++) {
		// First, find two smallest nodes 'min1, min2'
		// 每次先找到最小权值的两棵树
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			} else {
				min1i = pos2;
				pos2++;
			}
		} else {
			min1i = pos2;
			pos2++;
		}
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			} else {
				min2i = pos2;
				pos2++;
			}
		} else {
			min2i = pos2;
			pos2++;
		}
		//子树vocab_size+a为min1i和min2i的父亲节点
		//它的权是两个子树的权和
		count[vocab_size + a] = count[min1i] + count[min2i];
		//记录子树的父亲节点
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;
		//没有显示写出来的是binary[min1i]=0
		//也就是左子树编码0，右子树编码1
	}
	// Now assign binary code to each vocabulary word
	// huffman树已经构建完成，根据他为所有的词汇赋上编码code和辅助向量列point
	for (a = 0; a < vocab_size; a++) {
		b = a;
		//b表示词汇a所代表的叶子节点
		i = 0;
		//i记录code和point的长度
		while (1) {
			//利用parent_node的记录，从每一个叶子节点开始逆流而上
			//回溯找到一个词汇在huffman书中的路径，从而得到编码和辅助向量列
			//注意这样获取的code和point是倒序的
			//之后要倒过来赋给词汇vocab_word
			code[i] = binary[b];
			//code[i]容易理解，就是当前节点对应的0或者1
			point[i] = b;
			//b是从
			//point[i]就是当前节点的所在的位置，注意后面还要减掉vocab_size，保证从0开始到vocab_size-1
			i++;
			b = parent_node[b];
			//当b达到huffman树的顶部，回溯完毕跳出
			if (b == vocab_size * 2 - 2) break;
		}
		//code和point长度赋值
		//需要注意的是，code和point的数组容量是定长MAX_CODE_LENGTH
		vocab[a].codelen = i;
		//point[0]根本没有使用
		//注意，迭代中使用的辅助向量，从根节点0层一直用到倒数2层
		//而使用的编码(每个节点对应一个0或者1)，从1层用到最后1层
		vocab[a].point[0] = vocab_size - 2;
		for (b = 0; b < i; b++) {
			//注意这里的下标和倒过来赋值
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
	//内存释放
	//huffman树结构已经没有必要
	free(count);
	free(binary);
	free(parent_node);
}

//从训练文件里面获取词汇表
//同时获取训练文件的大小，这个大小用于并行计算时负载平衡
void LearnVocabFromTrainFile() {
	char word[MAX_STRING];
	FILE *fin;
	long long a, i;
	//词汇的存储使用了hash技术用于快速定位
	//首先初始化hash表
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	vocab_size = 0;
	//首先加入一个特殊词汇</s>，这个词表示换行符
	//用于分割行的标记，具体见ReadWord函数的注释
	AddWordToVocab((char *)"</s>");
	while (1) {
		//读入一个词汇
		//要注意一些约定
		//具体见函数注释
		ReadWord(word, fin);
		if (feof(fin)) break;
		//train_words是可以认为是所有词汇的数目
		//不过其实在后面的SortVocab里面，这个值将会重新计算
		//见SortVocab函数
		train_words++;
		if ((debug_mode > 1) && (train_words % 100000 == 0)) {
			printf("%lldK%c", train_words / 1000, 13);
			fflush(stdout);
		}
		//如果是新词汇，插入词汇表
		//否则，增加词汇频数
		i = SearchVocab(word);
		if (i == -1) {
			a = AddWordToVocab(word);
			vocab[a].cn = 1;
		} else vocab[i].cn++;
		//当词汇数目过大，将舍弃一些[到此时为止]频数小的词
		//具体见ReduceVocab函数
		if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
	}
	//词频从大到小重排Vocab，有副作用，具体见SortVocab函数
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	//获取train_file的size
	file_size = ftell(fin);
	fclose(fin);
}

//写入词汇到文件save_vocab_file
//格式
//词汇 词频\n
void SaveVocab() {
	long long i;
	FILE *fo = fopen(save_vocab_file, "wb");
	for (i = 0; i < vocab_size; i++) fprintf(fo, "%s %lld\n", vocab[i].word, vocab[i].cn);
	fclose(fo);
}

//从词汇文件中读取词汇
//构建词汇表和hash表
//同时获取训练文件的大小，这个大小用于并行计算时负载平衡
//而不是从训练文件中获取词汇
//词汇文件路径是read_vocab_file
//由用户指定，是一个c字符串，默认首字符"\0"
//词汇文件的格式应该是词汇，词汇频率
//具体来说
//词汇(词汇边界是换行符，空格，文件边界或者制表符号)
//词汇频数(一个整数，可以到long long的级别)
//一个分割符号，仅仅一个字符，通常为换行符
void ReadVocab() {
	long long a, i = 0;
	char c;
	char word[MAX_STRING];
	FILE *fin = fopen(read_vocab_file, "rb");
	if (fin == NULL) {
		printf("Vocabulary file not found\n");
		exit(1);
	}
	//vocab_hash顾名思义是词汇hash表，在程序一开始就已经分配好了内存
	//首先将之初始化
	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;
	vocab_size = 0;
	while (1) {
		//读取一个词汇
		//需要注意一个特殊的词汇</s>，也就是换行符
		//见下面这个函数的注释
		ReadWord(word, fin);
		if (feof(fin)) break;
		//一个词汇存储在vocab_word结构里面，结构的定义见代码开头
		//所有的vocab_word存储在一个线性词汇表格里面
		//为了方便词汇的查询使用了hash技术用于词汇定位
		//由于词汇表使用了hash技术
		//加入词汇到词汇表需要下面的函数
		a = AddWordToVocab(word);
		//下面读入词汇频率
		//同时由于fscanf和ReadWord的问题，我们要退掉词汇频数后面的一个表示边界的字符
		//因为否则fscanf不会读入这些符号，ReadWord将会读入
		fscanf(fin, "%lld%c", &vocab[a].cn, &c);
		i++;
	}
	//重排词汇表，有副作用，做了若干工作，具体见函数注释
	SortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", vocab_size);
		printf("Words in train file: %lld\n", train_words);
	}
	//获取训练文件的大小
	fin = fopen(train_file, "rb");
	if (fin == NULL) {
		printf("ERROR: training data file not found!\n");
		exit(1);
	}
	fseek(fin, 0, SEEK_END);
	file_size = ftell(fin);
	fclose(fin);
}

void InitNet() {
	long long a, b;
	//内存动态对齐
	//syn0就是我们最后计算输出的word对应的vector
	//layer1_size是每个vector的维度
	//128是所谓的alignment，要求为2的幂，对齐的结果，syn0分配的[地址]定是alignment和sizeof(void *)的公倍数
	//程序不透明的对齐大概是性能考虑，因为要求分配的内存块太大
	//返回值是0或者错误原因
	a = posix_memalign((void **)&syn0, 128, (long long)vocab_size * layer1_size * sizeof(real));
	if (syn0 == NULL) {printf("Memory allocation failed\n"); exit(1);}
	//参数hs表示是否使用hierarchical softmax，就是huffman树和逻辑回归结合的策略，具体见w2v原理
	if (hs) {
		//使用hs策略需要辅助向量
		//syn1是辅助向量集合，每个辅助向量的大小为layer1_size，辅助向量比syn0得数量(vocab_size)应该少一个
		//这里仍然用vocab_size会为下标的使用提供方便，具体见迭代学习过程
		a = posix_memalign((void **)&syn1, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (syn1 == NULL) {printf("Memory allocation failed\n"); exit(1);}
		//对辅助向量，直接用0初始化
		for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
			syn1[a * layer1_size + b] = 0;
	}
	//negartive表示是否使用随机负采样
	if (negative>0) {
		//使用负采样需要对每个词提供一个负采样向量
		a = posix_memalign((void **)&syn1neg, 128, (long long)vocab_size * layer1_size * sizeof(real));
		if (syn1neg == NULL) {printf("Memory allocation failed\n"); exit(1);}
		for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
			syn1neg[a * layer1_size + b] = 0;
	}
	//对所有的word对应向量，随机给出初始值
	//注意到代码调用rand之前从没有使用srand
	//也就是说，每次产生的随机序列都是一样的
	//另外，这里给出的初始值保证每个向量各分量正负均匀的落在(-0.5/单词维度)到(0.5/单词维度)两侧
	for (b = 0; b < layer1_size; b++) for (a = 0; a < vocab_size; a++)
		syn0[a * layer1_size + b] = (rand() / (real)RAND_MAX - 0.5) / layer1_size;
	//下面构建huffman树
	CreateBinaryTree();
}

//下面是w2v模型训练的主要过程
//
void *TrainModelThread(void *id) {
	long long a, b, d, word, last_word, sentence_length = 0, sentence_position = 0;
	//word是单词在词汇表里面的位置
	long long word_count = 0, last_word_count = 0, sen[MAX_SENTENCE_LENGTH + 1];
	//word_count统计学习了多少词汇，注意这个意思是，没学一个词汇样本，这个值就加一
	//last_word_count的作用是动态变化学习率
	//每学习完10000个样本，学习率就变小一次，直到学习率达到starting_alpha * 0.0001为止
	long long l1, l2, c, target, label;
	unsigned long long next_random = (long long)id;
	//随机数next_random，用固定的公式产生的
	//这里25214903917是一个大质数
	//next_random = next_random * (unsigned long long)25214903917 + 11;
	real f, g;
	clock_t now;
	//neu1是所有输入的上下文向量之和组成的向量，维数自然是layer1_size
	//这里neu1e是一个layer1_size维度的向量，表示一轮迭代后每一个word对应的vector的增量
	real *neu1 = (real *)calloc(layer1_size, sizeof(real));
	real *neu1e = (real *)calloc(layer1_size, sizeof(real));
	//训练语料分成num_threads份，并行计算
	FILE *fi = fopen(train_file, "rb");
	fseek(fi, file_size / (long long)num_threads * (long long)id, SEEK_SET);
	//train file is divided into num_threads parts, to calculate in multi-thread
	while (1) {
		//这个学习循环由3部分组成
		//1是学习率更新
		//2是新句子读取
		//3是根据一个单词和他的上下文进行模型学习
		//关于这部分学习，有一个问题，就是涉及到全局的syn0,syn1的读取和更新
		//为什么没有加锁？不会有问题么？

		//第一个部分开始
		if (word_count - last_word_count > 10000) {
			//这一部分学习率动态变化
			//当学完了10000个单词后
			//学习率就变小一次，直到学习率达到starting_alpha * 0.0001为止
			//here inside the if
			//if we have learned 10000 words
			//rejust the alpha value, which means the study rate
			word_count_actual += word_count - last_word_count;
			//到上次为止变化学习率为止学习单词数目
			last_word_count = word_count;
			if ((debug_mode > 1)) {
				now=clock();
				printf("%cAlpha: %f  Progress: %.2f%%  Words/thread/sec: %.2fk  ", 13, alpha,
						word_count_actual / (real)(train_words + 1) * 100,
						word_count_actual / ((real)(now - start + 1) / (real)CLOCKS_PER_SEC * 1000));
				fflush(stdout);
			}
			//学习率变化公式
			//分母+1保证变化率不为0
			alpha = starting_alpha * (1 - word_count_actual / (real)(train_words + 1));
			if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
		}
		//第一个部分结束

		//第二个部分开始
		if (sentence_length == 0) {
			//sentence_length==0表示开始一个新句子的训练
			//首先读入一整句，然后针对这个句子的每一个单词和上下文进行学习
			while (1) {
				word = ReadWordIndex(fi);
				// 从fin中读取一个单词，同时查找返回他在词汇表里的位置
				if (feof(fi)) break;
				if (word == -1) continue;
				//读到一个有效单词word_count就增加1
				word_count++;
				//word=0是</s>
				//这个词表示句子末端
				if (word == 0) break;
				// The subsampling randomly discards frequent words while keeping the ranking same
				// 是否使用了高频词亚采样
				if (sample > 0) {
					//ran的值，随着单词的[频率]增大而减少，随着sample的增大而减少
					//注意这里实际用的是频率，也就是(单词频数/总单词频数)
					//这里实际上有
					//ran=sqrt(s/f)+s/f
					//这里，s=sample，f=单词[频率]
					//高频词被舍弃的概率是1-ran
					real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
					next_random = next_random * (unsigned long long)25214903917 + 11;
					//下面吧next_random映射到0~1
					//于是高频词被舍弃的概率是1-ran
					if (ran < (next_random & 0xFFFF) / (real)65536) continue;
				}
				sen[sentence_length] = word;
				sentence_length++;
				//如果句子单词长度超过，则视为句子达到末端
				if (sentence_length >= MAX_SENTENCE_LENGTH) break;
			}
			sentence_position = 0;
		}
		//第二个部分结束

		//下面是学习一个单词及其上下文的第三部分
		if (feof(fi)) break;
		//学习单词数目达到本线程学习限度就停止学习
		//所以w2v基本上只对语料学习一次
		//那么，怎么保证收敛呢？
		if (word_count > train_words / num_threads) break;
		//刚刚读完句子sentence_position从0开始
		//然后每次循环一个单词
		//一个单词一个单词往下学习
		word = sen[sentence_position];
		//下面这句代码完全可以删掉，白白引起误解
		//作者本意是这个单词不是有效单词，不学习
		//但是这种情况不会发生，word==1根本不会被读入sen中，见上面读取句子的循环
		//另一方面，这种情况发生的话，sentence_position没有更新，将会在这里形成死循环
		if (word == -1) continue;
		//neu1是所有输入的上下文向量之和组成的向量，维数自然是layer1_size
		for (c = 0; c < layer1_size; c++) neu1[c] = 0;
		//这里neu1e是一个layer1_size维度的向量，表示一轮迭代后每一个word对应的vector的增量
		//每一个word对应的增量都是一样的，这一点通过求梯度可以证明
		for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
		//window，所谓的窗口，与一个单词上下文的取得有关，由用户指定，默认为5
		next_random = next_random * (unsigned long long)25214903917 + 11;
		//b是一个从0到window-1的数
		//我们选取的上下文就是下面这个连续的部分
		//(sentence_position- (window-b) ) ~ (sentence_position + (window-b) )
		//也就是不包含当前word的前后各window-b个单词，合在一起就是当前word的上下文
		b = next_random % window;
		//下面根据用户的选择有两种模型
		//cbow和skip-gram，关于这两种模型请参考作者论文
		//下面的步骤是随机梯度上升
		if (cbow) {  //train the cbow architecture
			//基于cbow模型的学习
			//首先获取word上下文,加和到neu1里
			//然后根据是否使用hs和负采样有不同的梯度公式
			//依次根据两种梯度公式更新辅助向量syn1
			//然后再更新word向量syn0
			
			// in -> hidden
			
			//我们选取的上下文就是下面这个连续的部分
			//(sentence_position- (window-b) ) ~ (sentence_position + (window-b) )
			//也就是不包含当前word的前后各window-b个单词，合在一起就是当前word的上下文
			for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
				c = sentence_position - window + a;
				//下面这两句话的意思是说，构成上下文时开头不足或者结尾不足就跳过
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				//下面这句可以删除
				if (last_word == -1) continue;
				//所有单词对应的vector各分量累加到neu1上
				//last_word是本次累加的word的index
				for (c = 0; c < layer1_size; c++) neu1[c] += syn0[c + last_word * layer1_size];
			}

			if (hs) for (d = 0; d < vocab[word].codelen; d++) {
				//如果使用hierarchical softmax
				//先更新辅助向量syn1
				//对应于一个word有codelen个辅助向量
				//根据梯度计算
				//辅助向量的更新一个一个来，互相没有影响
				//这个更新公式完整的由下面的代码来定义出来
				f = 0;
				//l2是对应word的第d个辅助向量的头位置
				l2 = vocab[word].point[d] * layer1_size;
				//f是当前(第d个)辅助向量与上下文单词向量和neu1的内积的sigmoid函数值
				// Propagate hidden -> output
				for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1[c + l2];
				// 下面这一部分应该注意
				// 根据表格查找sigmoid函数值
				// 但是f<=-MAX_EXP和f>=MAX_EXP的时候，为什么略过了呢？
				// 首先要明白，在-MAX_EXP以下，f近似为0
				// 在MAX_EXP以上，f近似为1
				// 作者的意思大概是，从逻辑回归二分的角度讲
				// 这个时候sigmoid的值近似于0和1，相应的code[d]则应该分别近似于1和0
				// 此时下面的g=0
				// 于是直接略去
				if (f <= -MAX_EXP) continue;
				else if (f >= MAX_EXP) continue;
				//否则查表得到sigmoid值
				else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
				// 'g' is the gradient multiplied by the learning rate
				// 下面的g*neu1正是当前辅助向量的梯度正方向*学习率
				// 而g*当前辅助向量则是在这个节点上word所对应的vector的梯度正方向*学习率
				// 需要把全部codelen个梯度方向加起来才是word对应vector的最后梯度正方向*学习率
				// 于是将他们累加到neu1e上
				g = (1 - vocab[word].code[d] - f) * alpha;
				// Propagate errors output -> hidden
				for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
				// Learn weights hidden -> output
				for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];
			}
			// NEGATIVE SAMPLING
			if (negative > 0) for (d = 0; d < negative + 1; d++) {
				//如果使用负采样
				//则对当前word进行negative次负采样
				//还有一次正采样
				//这个原理上比hs简单得多，保证概率最大的梯度方向即可
				//具体看作者论文
				if (d == 0) {
					//这一次是正采样
					target = word;
					label = 1;
				} else {
					//这下面是负采样的过程
					//差表格table
					//这样负采样的方法注释见InitUnigramTable函数定义
					next_random = next_random * (unsigned long long)25214903917 + 11;
					target = table[(next_random >> 16) % table_size];
					if (target == 0) target = next_random % (vocab_size - 1) + 1;
					if (target == word) continue;
					//如果负采样刚好采集到了正样本，那么不做负样本计算
					label = 0;
				}
				//请自行求导得到梯度公式后理解下面的部分，理解方法和cbow hs是一样的
				l2 = target * layer1_size;
				f = 0;
				for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
				if (f > MAX_EXP) g = (label - 1) * alpha;
				else if (f < -MAX_EXP) g = (label - 0) * alpha;
				else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
				for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
				for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
			}
			// hidden -> in
			
			// 最后neu1e更新到每一个输入的word
			// 这一部分代码只是平凡更新, 阅读参考上面取上下文部分
			for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
				c = sentence_position - window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				if (last_word == -1) continue;
				for (c = 0; c < layer1_size; c++) syn0[c + last_word * layer1_size] += neu1e[c];
			}
		} else {  //train skip-gram
			//使用skip-gram模型
			//本模型和上面的cbow模型基本一一对应
			//大家自行求梯度后阅读吧
			for (a = b; a < window * 2 + 1 - b; a++) if (a != window) {
				c = sentence_position - window + a;
				if (c < 0) continue;
				if (c >= sentence_length) continue;
				last_word = sen[c];
				if (last_word == -1) continue;
				l1 = last_word * layer1_size;
				for (c = 0; c < layer1_size; c++) neu1e[c] = 0;
				// HIERARCHICAL SOFTMAX
				if (hs) for (d = 0; d < vocab[word].codelen; d++) {
					f = 0;
					l2 = vocab[word].point[d] * layer1_size;
					// Propagate hidden -> output
					for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1[c + l2];
					if (f <= -MAX_EXP) continue;
					else if (f >= MAX_EXP) continue;
					else f = expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))];
					// 'g' is the gradient multiplied by the learning rate
					g = (1 - vocab[word].code[d] - f) * alpha;
					// Propagate errors output -> hidden
					for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1[c + l2];
					// Learn weights hidden -> output
					for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * syn0[c + l1];
				}
				// NEGATIVE SAMPLING
				if (negative > 0) for (d = 0; d < negative + 1; d++) {
					if (d == 0) {
						target = word;
						label = 1;
					} else {
						next_random = next_random * (unsigned long long)25214903917 + 11;
						target = table[(next_random >> 16) % table_size];
						if (target == 0) target = next_random % (vocab_size - 1) + 1;
						if (target == word) continue;
						label = 0;
					}
					l2 = target * layer1_size;
					f = 0;
					for (c = 0; c < layer1_size; c++) f += syn0[c + l1] * syn1neg[c + l2];
					if (f > MAX_EXP) g = (label - 1) * alpha;
					else if (f < -MAX_EXP) g = (label - 0) * alpha;
					else g = (label - expTable[(int)((f + MAX_EXP) * (EXP_TABLE_SIZE / MAX_EXP / 2))]) * alpha;
					for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
					for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * syn0[c + l1];
				}
				// Learn weights input -> hidden
				for (c = 0; c < layer1_size; c++) syn0[c + l1] += neu1e[c];
			}
		}
		//学习本句话下一个单词(以及其上下文)
		sentence_position++;
		if (sentence_position >= sentence_length) {
			//学完这一个句子的所有单词，进行下一个句子的学习
			sentence_length = 0;
			continue;
		}
	}
	//释放内存，结束进程
	fclose(fi);
	free(neu1);
	free(neu1e);
	pthread_exit(NULL);
}

void TrainModel() {
	long a, b, c, d;
	FILE *fo;
	pthread_t *pt = (pthread_t *)malloc(num_threads * sizeof(pthread_t));
	printf("Starting training using file %s\n", train_file);
	starting_alpha = alpha;
	//开始的学习率，由用户输入
	//如果用户没有指定，那么默认值0.025
	if (read_vocab_file[0] != 0) ReadVocab(); else LearnVocabFromTrainFile();
	//如果指定了词汇文件，那么读取词汇文件，不再从训练文件中获得词汇
	//否则从训练文件中获得词汇
	//如果指定了词汇文件，写入
	if (save_vocab_file[0] != 0) SaveVocab();
	//输出文件是必须指定的，输出内容是word和vector
	if (output_file[0] == 0) return;
	//初始化向量，辅助向量，负采样向量
	//构建huffman树
	//这个命名大概是觉得所有的向量内存开起来像个网格吧
	InitNet();
	//如果使用随机负采样
	//需要初始化一个配套table
	//具体见下面函数的定义
	if (negative > 0) InitUnigramTable();
	start = clock();
	//多线程训练
	for (a = 0; a < num_threads; a++) pthread_create(&pt[a], NULL, TrainModelThread, (void *)a);
	for (a = 0; a < num_threads; a++) pthread_join(pt[a], NULL);
	//训练完成
	fo = fopen(output_file, "wb");
	//classes表示是否进行聚类
	if (classes == 0) {
		//不进行k-means聚类
		//输出word2vector
		// Save the word vectors
		fprintf(fo, "%lld %lld\n", vocab_size, layer1_size);
		for (a = 0; a < vocab_size; a++) {
			fprintf(fo, "%s ", vocab[a].word);
			//binary表示是否用二进制格式输出
			if (binary) for (b = 0; b < layer1_size; b++) fwrite(&syn0[a * layer1_size + b], sizeof(real), 1, fo);
			else for (b = 0; b < layer1_size; b++) fprintf(fo, "%lf ", syn0[a * layer1_size + b]);
			fprintf(fo, "\n");
		}
	} else {
		//下面进行K-means，这一部分不做注释了
		// Run K-means on the word vectors
		int clcn = classes, iter = 10, closeid;
		int *centcn = (int *)malloc(classes * sizeof(int));
		int *cl = (int *)calloc(vocab_size, sizeof(int));
		real closev, x;
		real *cent = (real *)calloc(classes * layer1_size, sizeof(real));
		for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
		for (a = 0; a < iter; a++) {
			for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
			for (b = 0; b < clcn; b++) centcn[b] = 1;
			for (c = 0; c < vocab_size; c++) {
				for (d = 0; d < layer1_size; d++) {
					cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
					centcn[cl[c]]++;
				}
			}
			for (b = 0; b < clcn; b++) {
				closev = 0;
				for (c = 0; c < layer1_size; c++) {
					cent[layer1_size * b + c] /= centcn[b];
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}
				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
			}
			for (c = 0; c < vocab_size; c++) {
				closev = -10;
				closeid = 0;
				for (d = 0; d < clcn; d++) {
					x = 0;
					for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
					if (x > closev) {
						closev = x;
						closeid = d;
					}
				}
				cl[c] = closeid;
			}
		}
		// Save the K-means classes
		for (a = 0; a < vocab_size; a++) fprintf(fo, "%s %d\n", vocab[a].word, cl[a]);
		free(centcn);
		free(cent);
		free(cl);
	}
	fclose(fo);
}

int ArgPos(char *str, int argc, char **argv) {
	int a;
	for (a = 1; a < argc; a++) if (!strcmp(str, argv[a])) {
		if (a == argc - 1) {
			printf("Argument missing for %s\n", str);
			exit(1);
		}
		return a;
	}
	return -1;
}

int main(int argc, char **argv) {
	int i;
	if (argc == 1) {
		printf("WORD VECTOR estimation toolkit v 0.1b\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		//训练文本文件名
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		//输出文件名
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		//向量维数
		printf("\t-window <int>\n");
		printf("\t\tSet max skip length between words; default is 5\n");
		//所谓窗口数目，指的是上下文单词的个数
		printf("\t-sample <float>\n");
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency");
		//是否使用高频词亚采样，高频词将以一定的概率被舍弃掉
		printf(" in the training data will be randomly down-sampled; default is 0 (off), useful value is 1e-5\n");
		printf("\t-hs <int>\n");
		printf("\t\tUse Hierarchical Softmax; default is 1 (0 = not used)\n");
		//是否使用Hierarchical Softmax技术
		//所谓HS技术，是利用Huffman编码对训练算法的一种优化
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 0, common values are 5 - 10 (0 = not used)\n");
		//随机负采样的数目
		printf("\t-threads <int>\n");
		printf("\t\tUse <int> threads (default 1)\n");
		//进程数目
		printf("\t-min-count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; default is 5\n");
		//对低频训练词进行丢弃
		//这里设置丢弃词频阈值
		//默认是，词频5以下的词丢弃
		//虽然看起来很正常的策略，但是为什么好？
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025\n");
		//初始学习率
		//如代码中所述，默认值为0.025
		printf("\t-classes <int>\n");
		printf("\t\tOutput word classes rather than word vectors; default number of classes is 0 (vectors are written)\n");
		printf("\t-debug <int>\n");
		printf("\t\tSet the debug mode (default = 2 = more info during training)\n");
		printf("\t-binary <int>\n");
		printf("\t\tSave the resulting vectors in binary moded; default is 0 (off)\n");
		//是否以二进制的格式输出
		printf("\t-save-vocab <file>\n");
		printf("\t\tThe vocabulary will be saved to <file>\n");
		//词汇表存放文件
		printf("\t-read-vocab <file>\n");
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		//提供一个词汇文件，如果提供词汇文件，将使用这个文件中的词汇，不再通过训练数据得出词汇
		printf("\t-cbow <int>\n");
		printf("\t\tUse the continuous bag of words model; default is 0 (skip-gram model)\n");
		//是否使用所谓的CBOW模型，否则就使用所谓的skip-gram模型
		//这两种模型的讲解在后面的代码中进行
		printf("\nExamples:\n");
		printf("./word2vec -train data.txt -output vec.txt -debug 2 -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -binary 0 -cbow 1\n\n");
		return 0;
	}
	//获取所有文件参数之前，将其初始化
	output_file[0] = 0;
	save_vocab_file[0] = 0;
	read_vocab_file[0] = 0;
	if ((i = ArgPos((char *)"-size", argc, argv)) > 0) layer1_size = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-train", argc, argv)) > 0) strcpy(train_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-save-vocab", argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-read-vocab", argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-debug", argc, argv)) > 0) debug_mode = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-binary", argc, argv)) > 0) binary = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-cbow", argc, argv)) > 0) cbow = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-alpha", argc, argv)) > 0) alpha = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-output", argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	if ((i = ArgPos((char *)"-window", argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-sample", argc, argv)) > 0) sample = atof(argv[i + 1]);
	if ((i = ArgPos((char *)"-hs", argc, argv)) > 0) hs = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-negative", argc, argv)) > 0) negative = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-threads", argc, argv)) > 0) num_threads = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-min-count", argc, argv)) > 0) min_count = atoi(argv[i + 1]);
	if ((i = ArgPos((char *)"-classes", argc, argv)) > 0) classes = atoi(argv[i + 1]);
	vocab = (struct vocab_word *)calloc(vocab_max_size, sizeof(struct vocab_word));
	vocab_hash = (int *)calloc(vocab_hash_size, sizeof(int));

	//预先计算sigmoid函数
	//一个问题是，这里expTable没有被free？是我看错了么?
	//这里只计算了-EXP_MAX和EXP_MAX之间的函数值
	//小于-EXP_MAX的sigmoid值近似为0
	//大于EXP_MAX的近似为1
	//sigmoid=1/(1+exp(-x))=exp(x)/exp(x)+1
	expTable = (real *)malloc((EXP_TABLE_SIZE + 1) * sizeof(real));
	for (i = 0; i < EXP_TABLE_SIZE; i++) {
		expTable[i] = exp((i / (real)EXP_TABLE_SIZE * 2 - 1) * MAX_EXP); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1);                   // Precompute f(x) = x / (x + 1)
		// here we calculate the sigmoid function, from -6 to 6, (here MAX_EXP)
		// because when beyond (-6,6), the sigmoid is to be 0 and 1
		// split (-6,6) into 1000 segments
		// and calculate
		// sigmoid(x)=1/(1+e^(-x))

	}
	TrainModel();
	return 0;
}
