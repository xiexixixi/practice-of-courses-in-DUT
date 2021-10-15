#include<stdlib.h>
#include<stdio.h>
#include<windows.h>
#include<math.h>
#include<string.h>
#include"UI.h"
#include"translate.h"
#include"struct.h"
/*������д���ļ�����*/
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

/*linkid���������ַ�������*/
void linkid_binary_search(struct map *arr[], int length)
{
	unsigned int element;
	announce("������Ҫ���ҵ�linkid��");
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
		announce("δ������������\n");
	else if (result >= 0 && result<length)
	{
		show(arr[result]);
		//subAnnounce("");
		//show(arr[result]);
	}
	announce("�������\n");
}

/*·������*/
void roadname_search(struct map *arr[], int length)
{
	int flag = 0;
	char nam[30];
	announce("������Ҫ�����ĵ�·���ƣ�");
	scanf("%s", nam);
	putchar('\n');
	int i = 0/*j=0*/; 
	int num=0;
	for (i = 0; i<length; i++)
	{
		if (strcmp(nam, arr[i]->name) == 0 && arr[i]->roadnameflag == 1)//�Ա��ַ���
		{
			flag = 1;
			show(arr[i]);
			num++;
		}
	}
	if (flag == 0)
		announce("δ����������");
}

/*Class���ż���*/
void classnumber_search(struct map *arr[], int length)
{
	int classnum;
	announce("������Ҫ������Class���ţ�");
	scanf("%d", &classnum);
	putchar('\n');
	FILE *fp;                   //��ȡ�ļ�
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
		announce("δ�������÷���\n");
	else if (j>0 && j<3)
		announce("�������\n");
	else if (j<1)
		announce("\n�������\nȫ�������ѱ��洢��class_result.txt�ļ�\n");
	fclose(fp);
}

/*��·�ڼ���*/
void sideway_search(struct map *arr[], int length)
{
	int sidewaynum;
	announce("������Ҫ�����Ĳ�·����");
	scanf("%d", &sidewaynum);
	putchar('\n');
	FILE *fp;                   //��ȡ�ļ�
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
		announce("δ�������÷���\n");
	else if (j>0 && j<3)
		announce("�������\n");
	else if (j<1)
		announce("�������,ȫ�������ѱ��洢��sideway_result.txt�ļ�\n");

	fclose(fp);
}
