#include"UI.h"
float maxone(float *arr,int sz){//int型数组找最大值 
	int i=0;
	float Max=arr[0];
	for(i=0;i<sz-1;i++){
		if(arr[i+1]>arr[i])Max=arr[i+1];
	}
	return Max;
}

void histogram(int x,int y,int length,const char *A,float a, const char *B,float b, const char* C,float c, const char *D,float d, const char *E,float e){//柱状图 
	float data[5]={a,b,c,d,e};
	const char *name[5]={A,B,C,D,E};
	float Max=maxone(data,5);//最大值 
	int i=0;//控制循环
	float ratio=0;//百分比 
	for(i=0;i<length+16;i++){
		gotoxyAndPutchar(x+i,y-1,"-",15);
		gotoxyAndPutchar(x+i,y+13,"-",15);
	} 
	for(i=0;i<13;i++){
    	gotoxyAndPutchar(x+6,y+i,"|",15);
    	if(i%3==0){gotoxyAndPutchar(x,y+i,name[i/3],12);} 
    }
    
	for(i=0;i<5;i++){
		ratio=((float)data[i]/Max);
		int j=0;//显示 
		for(j=0;j<length*ratio;j++){
			gotoxyAndPutchar(x+7+j,y+i*3,"#",10);
		} 
		gotoxy(x+7+j,y+i*3); printf("  %.2fs",data[i]);
	} 
	gotoxy(x,y+14);
	
}

