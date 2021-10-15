#include<stdlib.h>
#include<stdio.h>
#include<windows.h>
#include<math.h>
#include<string.h>
#include<iostream>
#include"struct.h"
//���뺯��������ȡ����ת��Ϊmap�ṹ�壩 
void translate(struct map *MAP, FILE	*fp) {
	unsigned short datasize = 0;    //���ڼ������ʱ����ı��� 
	unsigned linkid = 0;
	unsigned short roadnamesize = 0;
	unsigned short roadnameflag = 0;
	unsigned short branch = 0;
	unsigned short dispclass = 0;
	char name[30] = {
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
	};

	int i;//ѭ������ 

		  //ʹ�ý���ת������ȡ���� 
	for (i = 1; i >= 0; i--) { datasize += (fgetc(fp)*pow(256, i)); }
	for (i = 3; i >= 0; i--) { linkid += (fgetc(fp)*pow(256, i)); }
	for (i = 1; i >= 0; i--) { roadnamesize += fgetc(fp)*pow(256, i); }

	fgetc(fp);//��������reserved�ֽ� 
	fgetc(fp);
	fgetc(fp);

	int a;       //a���ڴ洢nodeԭʼ�鱨
	a = fgetc(fp);
	int s[8] = { 0,0,0,0,0,0,0,0 }, rem;//s���ڴ�Ŷ����ƻ������ 


	i = 0;//������������ 
	while (a) { rem = a % 2; a = a / 2; s[i++] = rem; }//a�����ƻ����Ž�s 

	dispclass = s[3] * 8 + s[2] * 4 + s[1] * 2 + s[0];//��s���н����õ����� 
	branch = s[6] * 4 + s[5] * 2 + s[4];
	roadnameflag = s[7];

	MAP->datasize = datasize;     //�洢�����õ������� 
	MAP->linkid = linkid;
	MAP->roadnameflag = roadnameflag;
	MAP->branch = branch;
	MAP->dispclass = dispclass;

	if (datasize != 12) { //�洢name 
		for (i = 0; i<datasize - 12; i++) {
			name[i] = fgetc(fp);
		}
		name[i] = '\0';
		strcpy(MAP->name, name);
	}
}

struct map* GenerateMAP(FILE *fp) {        //�ṹ��ָ���������� 
	struct map *map_origin = (struct map *)malloc(sizeof(struct map));
	translate(map_origin, fp);
	return map_origin;
}

void show(struct map *MAP) {             //�ṹ����Ϣ������� 
	//printf("    ");
	printf("datasize:%d", MAP->datasize); printf("	");
	printf("linkid:%d", MAP->linkid); printf("	");
	printf("flag:%d", MAP->roadnameflag); printf("	");
	printf("branch:%d", MAP->branch); printf("	");
	printf("dispclass:%d", MAP->dispclass); printf("	");
	if (MAP->datasize - 12 != 0) {
		printf("name:%s", MAP->name);
	}
	printf("\n");
}
