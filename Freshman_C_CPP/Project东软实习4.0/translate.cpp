#include<stdlib.h>
#include<stdio.h>
#include<windows.h>
#include<math.h>
#include<string.h>
#include<iostream>
#include"struct.h"
//翻译函数（将读取内容转化为map结构体） 
void translate(struct map *MAP, FILE	*fp) {
	unsigned short datasize = 0;    //用于计算和临时储存的变量 
	unsigned linkid = 0;
	unsigned short roadnamesize = 0;
	unsigned short roadnameflag = 0;
	unsigned short branch = 0;
	unsigned short dispclass = 0;
	char name[30] = {
		0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
	};

	int i;//循环变量 

		  //使用进制转换，读取内容 
	for (i = 1; i >= 0; i--) { datasize += (fgetc(fp)*pow(256, i)); }
	for (i = 3; i >= 0; i--) { linkid += (fgetc(fp)*pow(256, i)); }
	for (i = 1; i >= 0; i--) { roadnamesize += fgetc(fp)*pow(256, i); }

	fgetc(fp);//跳过三个reserved字节 
	fgetc(fp);
	fgetc(fp);

	int a;       //a用于存储node原始情报
	a = fgetc(fp);
	int s[8] = { 0,0,0,0,0,0,0,0 }, rem;//s用于存放二进制化后的数 


	i = 0;//计数变量清零 
	while (a) { rem = a % 2; a = a / 2; s[i++] = rem; }//a二进制化后存放进s 

	dispclass = s[3] * 8 + s[2] * 4 + s[1] * 2 + s[0];//对s进行解析得到数据 
	branch = s[6] * 4 + s[5] * 2 + s[4];
	roadnameflag = s[7];

	MAP->datasize = datasize;     //存储解析得到的数据 
	MAP->linkid = linkid;
	MAP->roadnameflag = roadnameflag;
	MAP->branch = branch;
	MAP->dispclass = dispclass;

	if (datasize != 12) { //存储name 
		for (i = 0; i<datasize - 12; i++) {
			name[i] = fgetc(fp);
		}
		name[i] = '\0';
		strcpy(MAP->name, name);
	}
}

struct map* GenerateMAP(FILE *fp) {        //结构体指针生产函数 
	struct map *map_origin = (struct map *)malloc(sizeof(struct map));
	translate(map_origin, fp);
	return map_origin;
}

void show(struct map *MAP) {             //结构体信息输出函数 
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
