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
		case '1':announce("���ڶ�ȡ�ļ�����");
			FILE *fp;//��ȡ�ļ�
			struct map *pool[63537];//�������ṹ��ָ������pool������ 
			if ((fp = fopen("GTBL.dat", "rb")) != NULL)//����ļ���ȡ�Ƿ�ɹ�
			{
				for (i = 0; i < 63537 && !feof(fp); i++)
				{
					pool[i] = GenerateMAP(fp);//���ö�ȡ����
				}
				fclose(fp);
				printf("��ȡ�ɹ���");
			}
			else
			{
				printf("��ȡʧ��");
				exit(1);
			}
			break;
		case '2':announce("�������򡭡����Ժ�\n");
			for (int aa = 1; aa < 6; aa++)
			{
				Sort(pool, 63537, aa);//���ε���������
				if ((fp = fopen("GTBL.dat", "rb")) != NULL)//����ļ���ȡ�Ƿ�ɹ�
				{
					for (i = 0; i < 63537 && !feof(fp); i++)
					{
						pool[i] = GenerateMAP(fp);//���ö�ȡ����
					}
					fclose(fp);
				}
			}
			quickSort(pool, 0, 63536);
			flag = 1;
			/*
			for (i = 0; i < 63537 && !feof(fp); i++)
			{
				show(pool[i]);//�����������������
			}
			printf("����Ϊ�����Ľ�������Ŀ��");
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
			case '1':if (flag - 1)//����Ƿ��Ѿ�����
			{
				shellSort(pool, 63536);//���������� 
			}
					 linkid_binary_search(pool, 63537);//���ö��ַ���������
					 break;
			case '2':classnumber_search(pool, 63537);//���÷��ż�������
				break;
			case '3':sideway_search(pool, 63537);//���ò�·����������
				break;
			case '4':roadname_search(pool, 63537);//���õ�·���Ƽ�������
				break;
			default:exit(1);
			}
			printf("\n\n\n\n\n\n\n                              ");
			system("pause");
			system("cls");
			break;
		case '4':announce("�����С���");
			if ((fp = fopen("GTBL.dat", "rb")) == NULL) {
				printf("error"); system("pause");
			}
			update(fp);			//���ø��º��� 
			printf("������ϣ�");
			break;
		case '0':exit(0);
		default:exit(1);
		}
	}
	system("pause");
	return 0;
}