#include <iostream>
#include <cstdlib>
#include <cstdio>
#include<fstream>

#include <string>  
#include <ctime>  
#include <string.h>  
#include <io.h>
#include <queue>


#define _CRT_SECURE_NO_WARNINGS

const unsigned N = 0.5e7;
//const unsigned N = 100;
const int Batch = 10;
const int One_batch = 1e6;
int counter = 0;
const int bias = 2e1+0.34;
using namespace std;


void show_arr(int* a, int num = N)
{
	cout << endl << "�����" << endl;
	for (int i = 0; i < num; i++)
		cout << a[i] << "\t";
}




string data_generator(int seq,string post = ".dat",int seed = time(0))//����һ��datax.dat�ļ�������Ϊint������N��
{
	int* one_patch = new int[N];

	srand((unsigned)seed);
	for (int i = 0; i < N; i++)
		one_patch[i] = rand();

	ofstream outfile;
	string filename = "data" + to_string(seq) + post;
	outfile.open(filename, ios::out | ios::trunc);

	for (int i = 0; i < N; i++)
		outfile << one_patch[i] << ' ';
	outfile.close();
	delete[] one_patch;

	cout << "����" + filename << endl;
	return filename;
}

void read_to_mem(string filename, int* mem,int batch = N / Batch)
{
	fstream infile;
	infile.open(filename, ios::in);

	for (int i = 0; i < batch; i++)
		infile >> mem[i];
	infile.close();

}

void write_to_file(string filename, int* mem,int app = 0)
{
	ofstream outfile;
	if(app)
		outfile.open(filename, ios::out | ios::app);
	else
		outfile.open(filename, ios::out | ios::trunc);
	for (int i = 0; i < N/Batch; i++)
		outfile << mem[i] << ' ';

}

void quick_sort(int* a, int left, int right) //left��rightΪ����ֵ
{
	int temp; //�洢ÿ��ѡ���Ļ�׼�����������ѡ��׼����
	int t;
	int initial = left;
	int end = right;
	temp = a[left];

	if (left > right)  
		return;

	while (left != right) 
	{
		while (a[right] >= temp && left < right)  //ֱ���ҵ�С�ڻ�׼����ֵΪ׼
			right--;
		while (a[left] <= temp && left < right)
			left++;
		if (left < right)  //������������ֵ����left=rightʱ���������whileѭ��
		{
			t = a[right];
			a[right] = a[left];
			a[left] = t;
		}
	}
	a[initial] = a[left];
	a[left] = temp;        //������λ

	//�ݹ鴦���λ��Ļ�׼������������
	quick_sort(a, initial, left - 1);  //��ʱleft=right
	quick_sort(a, left + 1, end);
}

string sort_file(string filename)
{
	clock_t starttime, endtime;

	int* get_data = new int[N / Batch];
	read_to_mem(filename, get_data);
	starttime = clock();
	quick_sort(get_data, 0, N / Batch - 1);
	endtime = clock();
	write_to_file(filename, get_data);
	cout << "������ʱ��" << bias*(double)(endtime - starttime) / CLOCKS_PER_SEC << "s" << endl;
	return filename;
}


string Merge_2_files(string file1, string file2)
{
	clock_t starttime, endtime;
	cout<<"�ϲ�" << file1<<" �� " << file2 << "\t";
	int* mem1 = new int[One_batch];
	int* mem2 = new int[One_batch];


	string des = "data" + to_string(counter++) + ".txt";

	fstream infile1;
	infile1.open(file1, ios::in);
	fstream infile2;
	infile2.open(file2, ios::in);

	fstream outfile;
	outfile.open(des, ios::out | ios::app);
	outfile.clear();
	int* d = new int[2 * One_batch];

	starttime = clock();

	while (!infile1.eof())
	{
		for (int i = 0; i < One_batch; i++)
		{
			if(!infile1.eof())
				infile1 >> mem1[i];
		}

		for (int i = 0; i < One_batch; i++)
			if (!infile1.eof())
				infile2 >> mem2[i];

		int i = 0,j=0,k=0;
		while (i< One_batch && j < One_batch)
		{
			if (mem1[i] < mem2[j])
				d[k++] = mem1[i++];
			else
				d[k++] = mem2[j++];
		}
		while (i< One_batch)
			d[k++] = mem1[i++];
		while (i< One_batch)
			d[k++] = mem2[j++];
		write_to_file(des, d,1);
	}

	infile1.close();
	infile2.close();
	outfile.close();

	endtime = clock();

	cout << "��ʱ: " << bias *(double)(endtime - starttime) / CLOCKS_PER_SEC << "s" << endl;

	return des;

}


int main()
{
	//�������ںϲ��ļ�
	queue<string> qfile;
	string filename;

	cout << "��" << 1000000000 << "���������ݣ���Ϊ" << Batch << "���ļ�" << endl;
	//����+�����ļ�
	for (counter = 0; counter < Batch; counter++)
	{
		srand((unsigned)time(0));
		filename = data_generator(counter,".txt");
		qfile.push(sort_file(filename));
	}

	while (qfile.size() != 1)
	{
		string file1 = qfile.front();
		qfile.pop();
		string file2 = qfile.front();
		qfile.pop();


		qfile.push(Merge_2_files(file1, file2));

	}


	return 0;
}


