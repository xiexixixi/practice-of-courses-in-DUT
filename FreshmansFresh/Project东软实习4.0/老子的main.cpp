#include<stdlib.h>
#include<stdio.h>
#include<windows.h>
#include<math.h>
#include<string.h>
#include<conio.h>
#include "sort.h"
#include"translate.h"
#include"UI.h"
#include"search.h"
#include"struct.h"
#include"update.h"
int main()
{
	system("mode con cols=100 lines=34");
	hideCursor();
	short int flag = 0;
	unsigned int i = 0;
	showboard();
	while (1)
	{
		int a;
		basicInterface();
		a = getchar();
		getchar();
		switch (a)
		{
		case '1':announce("正在读取文件……");
			FILE *fp;//读取文件
			struct map *pool[63537];//》》》结构体指针数组pool《《《 
			if ((fp = fopen("GTBL.dat", "rb")) != NULL)//检测文件读取是否成功
			{
				for (i = 0; i < 63537 && !feof(fp); i++)
				{
					pool[i] = GenerateMAP(fp);//调用读取函数
				}
				fclose(fp);
				printf("读取成功！");
			}
			else
			{
				printf("读取失败");
				exit(1);
			}
			break;
		case '2':announce("正在排序……请稍候\n");
			for (int aa = 1; aa < 6; aa++)
			{
				Sort(pool, 63537, aa);//依次调用排序函数
				if ((fp = fopen("GTBL.dat", "rb")) != NULL)//检测文件读取是否成功
				{
					for (i = 0; i < 63537 && !feof(fp); i++)
					{
						pool[i] = GenerateMAP(fp);//调用读取函数
					}
					fclose(fp);
				}
			}
			quickSort(pool, 0, 63536);
			flag = 1;
			/*
			for (i = 0; i < 63537 && !feof(fp); i++)
			{
				show(pool[i]);//调用输出排序结果函数
			}
			printf("以上为排序后的结果，请过目。");
			system("pause");
			system("cls");
			basicInterface();
			announce("");*/
			break;
		case '3':
			system("cls");
			searchInterface();
			int b;
			b = getch();
			switch (b)
			{
			case '1':if (flag - 1)//检测是否已经排序
			{
				shellSort(pool, 63536);//调用排序函数 
			}
					 linkid_binary_search(pool, 63537);//调用二分法检索函数
					 break;
			case '2':classnumber_search(pool, 63537);//调用番号检索函数
				break;
			case '3':sideway_search(pool, 63537);//调用岔路数检索函数
				break;
			case '4':roadname_search(pool, 63537);//调用道路名称检索函数
				break;
			default:exit(1);
			}
			printf("\n\n\n\n\n\n\n                              ");
			system("pause");
			system("cls");
			break;
		case '4':announce("更新中……");
			if ((fp = fopen("GTBL.dat", "rb")) == NULL) {
				printf("error"); system("pause");
			}
			update(fp);			//调用更新函数 
			printf("更新完毕！");
			break;
		case '0':exit(0);
		default:exit(1);
		}
	}
	system("pause");
	return 0;
}