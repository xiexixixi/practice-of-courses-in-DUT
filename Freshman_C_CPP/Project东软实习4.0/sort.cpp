#include "sort.h"
#include<time.h>
#include <iostream>
#include <stack>
#include"UI.h"
#include"struct.h"
using namespace std;
int partition(struct map *a[], int low, int high)
{
	struct map *p = a[low];
	while (low<high)
	{
		while (low<high && a[high]->linkid >= p->linkid) high--;
		a[low] = a[high];
		while (low<high && a[low]->linkid <= p->linkid) low++;
		a[high] = a[low];
	}
	a[low] = p;
	return low;
}
/*void quickSort(vector<int> &v, int left, int right)
{
	if (left >= right)
		return;
	stack<int> s;
	s.push(left);
	s.push(right);

	while (!s.empty())
	{
		int right = s.top();
		s.pop();
		int left = s.top();
		s.pop();
		if (left < right)
		{
			int boundary = partition1(v, left, right);
			// ������
			s.push(left);
			s.push(boundary);
			// ������
			s.push(boundary + 1);
			s.push(right);
		}

	}
}*/
void quickSort(struct map *a[], int p, int q)
{
	stack<int> st;
	int j;
	do {
		while (p<q)
		{
			j = partition(a, p, q);
			if ((j - p)<(q - j))
			{
				st.push(j + 1);
				st.push(q);
				q = j - 1;
			}
			else
			{
				st.push(p);
				st.push(j - 1);
				p = j + 1;
			}
		}
		if (st.empty()) return;
		q = st.top(); st.pop();
		p = st.top(); st.pop();
	} while (1);
}
/*void quickSort(map * mapArr[], int left, int right)
{
	if (left< right)
	{
		int i = left, j = right;
		unsigned x = mapArr[left]->linkid;
		map * ptemp = mapArr[left];
		while (i < j)
		{
			while (i < j && mapArr[j]->linkid >= x) // ���������ҵ�һ��С��x����
				j--;
			if (i < j)
				mapArr[i++] = mapArr[j];
			while (i < j && mapArr[i]->linkid< x) // ���������ҵ�һ�����ڵ���x����
				i++;
			if (i < j)
				mapArr[j--] = mapArr[i];
		}
		mapArr[i] = ptemp;
		quickSort(mapArr, left, i - 1); // �ݹ����
		quickSort(mapArr, i + 1, right);
	}
}*/
map * bubbleSort(map * mapArr[], unsigned int n)
{
	map * ptemp;
	for (int i = 0; i < n; i++)
		for (int j = 1; j < n - i; j++)
		{
			if (mapArr[j]->linkid < mapArr[j - 1]->linkid)
			{
				ptemp = mapArr[j];
				mapArr[j] = mapArr[j - 1];
				mapArr[j - 1] = ptemp;
			}
		}
	return *mapArr;
}
void selectSort(struct map **x, int n)
{
	int i, j, min;
	struct map *t;
	for (i = 0; i<n - 1; i++) /*Ҫѡ��Ĵ�����0~n-2 ��n-1 ��*/
	{
		min = i; /*���赱ǰ�±�Ϊi ������С���ȽϺ��ٵ���*/
		for (j = i + 1; j<n; j++)/*ѭ���ҳ���С�������±����ĸ�*/
		{
			if (x[j]->linkid<x[min]->linkid)
			{
				min = j; /*������������ǰ�����������������±�*/
			}
		}
		if (min != i) /*���min ��ѭ���иı��ˣ�����Ҫ��������*/
		{
			t = x[i];
			x[i] = x[min];
			x[min] = t;
		}
	}
}
void insertSort(struct map **x, int n)
{
	int i, j;
	struct map * t;
	for (i = 1; i<n; i++)
	{
		t = x[i];
		j = i - 1;
		while (j >= 0 && t->linkid<x[j]->linkid)
		{
			x[j + 1] = x[j];
			j--;
		}
		x[j + 1] = t;
	}
}
void shellSort(struct map * pool[], int length)
{
	int increment;
	int i, j;
	struct map *temp;
	for (increment = length / 2; increment > 0; increment /= 2) //�������Ʋ���,���ݼ���1
	{ 		// i�ӵ�step��ʼ���У�ӦΪ��������ĵ�һ��Ԫ��
			// �����Ȳ������ӵڶ�����ʼ����
		for (i = increment; i < length; i++)
		{
			temp = pool[i];
			for (j = i - increment; j >= 0 && (temp->linkid)< (pool[j]->linkid); j -= increment)
			{
				pool[j + increment] = pool[j];
			}
			pool[j + increment] = temp; //����һ��λ������
		}
	}
}
void Sort(struct map * mapArr[], int n, short mode)
{
	double start, end;
	start = clock();
	switch (mode)
	{
	case 1:
		bubbleSort(mapArr, n);
		end = clock();
		printf("bubbleSort��ʱ��%lf seconds\n", (end - start) / CLOCKS_PER_SEC);
		break;
	case 2:
		insertSort(mapArr, n);
		end = clock();
		printf("insertSort��ʱ��%lf seconds\n", (end - start) / CLOCKS_PER_SEC);
		break;
	case 3:
		shellSort(mapArr, n);
		end = clock();
		printf("shellSort��ʱ��%lf seconds\n", (end - start) / CLOCKS_PER_SEC);
		break;
	case 4:
		selectSort(mapArr, n);
		end = clock();
		printf("selectSort��ʱ��%lf seconds\n", (end - start) / CLOCKS_PER_SEC);
		break;
	case 5:
		quickSort(mapArr, 0, n - 1);
		end = clock();
		printf("quickSort��ʱ��%lf seconds\n", (end - start) / CLOCKS_PER_SEC);
		break;
	default:
		exit(1);
		break;
	}
}