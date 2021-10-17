#include<stdlib.h>
#include<stdio.h>
#include<windows.h>
#include<math.h>
#include<string.h>
#include"UI.h"
#include"translate.h"
#include"struct.h"
/*将数据写入文件函数*/
void write(struct map *arr, FILE* fp)
{
	char str[15];
	itoa(arr->datasize, str, 10);
	fputs(str, fp); fputc(' ', fp);
	itoa(arr->linkid, str, 10);
	fputs(str, fp); fputc(' ', fp);
	itoa(arr->roadnameflag, str, 10);
	fputs(str, fp); fputc(' ', fp);
	itoa(arr->branch, str, 10);
	fputs(str, fp); fputc(' ', fp);
	itoa(arr->dispclass, str, 10);
	fputs(str, fp); fputc(' ', fp);
	if (arr->datasize - 12 != 0) {
		fputs(arr->name, fp);
	}
	fputc('\n', fp);
}

/*linkid检索（二分法检索）*/
void linkid_binary_search(struct map *arr[], int length)
{
	unsigned int element;
	announce("请输入要查找的linkid：");
	scanf("%d", &element);
	putchar('\n');
	int result = -1;
	int left = 0, right = length;
	while (left <= right)
	{
		int mid = (left + right) / 2;
		if ((arr[mid]->linkid)>element)
		{
			right = mid - 1;
		}
		else if ((arr[mid]->linkid)<element)
		{
			left = mid + 1;
		}
		else
		{
			result = mid;
			break;
		}
	}
	if (result == -1)
		announce("未检索到该数据\n");
	else if (result >= 0 && result<length)
	{
		show(arr[result]);
		//subAnnounce("");
		//show(arr[result]);
	}
	announce("检索完毕\n");
}

/*路名检索*/
void roadname_search(struct map *arr[], int length)
{
	int flag = 0;
	char nam[30];
	announce("请输入要检索的道路名称：");
	scanf("%s", nam);
	putchar('\n');
	int i = 0/*j=0*/; 
	int num=0;
	for (i = 0; i<length; i++)
	{
		if (strcmp(nam, arr[i]->name) == 0 && arr[i]->roadnameflag == 1)//对比字符串
		{
			flag = 1;
			show(arr[i]);
			num++;
		}
	}
	if (flag == 0)
		announce("未检索到数据");
}

/*Class番号检索*/
void classnumber_search(struct map *arr[], int length)
{
	int classnum;
	announce("请输入要检索的Class番号：");
	scanf("%d", &classnum);
	putchar('\n');
	FILE *fp;                   //读取文件
	if ((fp = fopen("class_result.txt", "w+")) == NULL) {
		printf("open failed");
		system("pause");
	}
	int i = 0, j = 3;
	for (i = 0; i<length; i++)
	{
		if ((arr[i]->dispclass) == classnum)
		{
			j--;
			if (j >= 0)
			{
				show(arr[i]);
				putchar('\n');
			}
			write(arr[i], fp);
		}
	}
	if (j == 3)
		announce("未检索到该番号\n");
	else if (j>0 && j<3)
		announce("检索完毕\n");
	else if (j<1)
		announce("\n检索完毕\n全部数据已被存储至class_result.txt文件\n");
	fclose(fp);
}

/*岔路口检索*/
void sideway_search(struct map *arr[], int length)
{
	int sidewaynum;
	announce("请输入要检索的岔路数：");
	scanf("%d", &sidewaynum);
	putchar('\n');
	FILE *fp;                   //读取文件
	if ((fp = fopen("sideway_result.txt", "w+")) == NULL) {
		printf("open failed");
		system("pause");
	}
	int i = 0, j = 3;
	for (i = 0; i<length; i++)
	{
		if (arr[i]->branch == sidewaynum)
		{
			j--;
			if (j >= 0)
			{
				show(arr[i]);
				putchar('\n');
			}
			write(arr[i], fp);
		}
	}
	if (j == 3)
		announce("未检索到该番号\n");
	else if (j>0 && j<3)
		announce("检索完毕\n");
	else if (j<1)
		announce("检索完毕,全部数据已被存储至sideway_result.txt文件\n");

	fclose(fp);
}
