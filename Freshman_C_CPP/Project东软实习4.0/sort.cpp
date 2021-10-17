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
			// 左区间
			s.push(left);
			s.push(boundary);
			// 右区间
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
			while (i < j && mapArr[j]->linkid >= x) // 从右向左找第一个小于x的数
				j--;
			if (i < j)
				mapArr[i++] = mapArr[j];
			while (i < j && mapArr[i]->linkid< x) // 从左向右找第一个大于等于x的数
				i++;
			if (i < j)
				mapArr[j--] = mapArr[i];
		}
		mapArr[i] = ptemp;
		quickSort(mapArr, left, i - 1); // 递归调用
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
	for (i = 0; i<n - 1; i++) /*要选择的次数：0~n-2 共n-1 次*/
	{
		min = i; /*假设当前下标为i 的数最小，比较后再调整*/
		for (j = i + 1; j<n; j++)/*循环找出最小的数的下标是哪个*/
		{
			if (x[j]->linkid<x[min]->linkid)
			{
				min = j; /*如果后面的数比前面的数大，则记下它的下标*/
			}
		}
		if (min != i) /*如果min 在循环中改变了，就需要交换数据*/
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
	for (increment = length / 2; increment > 0; increment /= 2) //用来控制步长,最后递减到1
	{ 		// i从第step开始排列，应为插入排序的第一个元素
			// 可以先不动，从第二个开始排序
		for (i = increment; i < length; i++)
		{
			temp = pool[i];
			for (j = i - increment; j >= 0 && (temp->linkid)< (pool[j]->linkid); j -= increment)
			{
				pool[j + increment] = pool[j];
			}
			pool[j + increment] = temp; //将第一个位置填上
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
		printf("bubbleSort用时：%lf seconds\n", (end - start) / CLOCKS_PER_SEC);
		break;
	case 2:
		insertSort(mapArr, n);
		end = clock();
		printf("insertSort用时：%lf seconds\n", (end - start) / CLOCKS_PER_SEC);
		break;
	case 3:
		shellSort(mapArr, n);
		end = clock();
		printf("shellSort用时：%lf seconds\n", (end - start) / CLOCKS_PER_SEC);
		break;
	case 4:
		selectSort(mapArr, n);
		end = clock();
		printf("selectSort用时：%lf seconds\n", (end - start) / CLOCKS_PER_SEC);
		break;
	case 5:
		quickSort(mapArr, 0, n - 1);
		end = clock();
		printf("quickSort用时：%lf seconds\n", (end - start) / CLOCKS_PER_SEC);
		break;
	default:
		exit(1);
		break;
	}
}