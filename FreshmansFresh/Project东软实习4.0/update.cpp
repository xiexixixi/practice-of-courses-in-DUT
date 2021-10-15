#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"update.h"

int  refind(FILE *fp, int whatyouwant, int *longsum) {//从头开始的fp指针 
	int ilong = 0;
	int iid = 0;
	int i = 0;
	for (i = 1; i >= 0; i--) { ilong += ((int)fgetc(fp)*pow(256, i)); }
	for (i = 3; i >= 0; i--) { iid += ((int)fgetc(fp)*pow(256, i)); }
	if (iid == whatyouwant) {
		return 1;
	}
	else {
		for (i = 0; i<ilong - 6; i++) {
			fgetc(fp);
		}
		*longsum += ilong;
		return 0;
	}
}
void renew(int longsum, FILE *fp, int flag, int bruch, int dispclass) {//ilong为上一个函数的返回值 
	int sum = flag * 128 + bruch * 16 + dispclass;
	fseek(fp, longsum, 0); fseek(fp, 11, 1);
	printf("\n\n\n%d", fwrite(&sum, 1, 1, fp));//fflush(fp);
	int i = 0;
	/*for(i=12;i<ilong;i++){
	fwrite(s+i,1,1,fp);fflush(fp);
	}*/
}
void reNew(FILE *Fp,int whatidyouwant, int flag, int bruch, int dispclass) {
	if (flag<0 || flag>1 || bruch<0 || bruch>7 || dispclass<0 || dispclass>15) {
		printf("数据有误"); system("pause");
	}
	if ((Fp = fopen("GTBL.dat", "rb+")) == NULL) {
		printf("open failed");
		system("pause");
	}

	int longsum = 0;
	int i = 0;
	for (i = 0; i<63537; i++) {
		if (refind(Fp, whatidyouwant, &longsum)) {
			renew(longsum, Fp, flag, bruch, dispclass);
			break;
		}
		else {
			continue;
		}
	}
	//fclose(Fp);
}
void update(FILE *fp) 
{
	int flag = 0, bruch = 0, dispclass = 0, searchid = 0;
	printf("input id:");
	scanf("%d", &searchid);
	printf("input new flag[0,1]:");
	scanf("%d", &flag);
	printf("input new bruch[0,7]:");
	scanf("%d", &bruch);
	printf("input new dispclass[0,15]:");
	scanf("%d", &dispclass);
	reNew(fp,searchid, flag, bruch, dispclass);
}