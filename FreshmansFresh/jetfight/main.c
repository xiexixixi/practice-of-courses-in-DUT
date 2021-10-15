#include<stdio.h>
#include<windows.h>
#include <stdlib.h>
#include <time.h>
#include <conio.h>
#define H 101
#define W 101
#define path "D:\\log.txt"
//�ո�0���ӵ�1���ҷ��ɻ�2���з��ɻ�4
static int scr[101][101]={0},h=40,w=60,plh,plw,density=300,speed=100,life,t=0,score,lastscr[101][101]={0};
HANDLE hOutput;
int change(int i,int j)//�ж�λ�øı�
{
	if(scr[i][j]!=lastscr[i][j])
	{
		lastscr[i][j]=scr[i][j];
		return 1;
	}
	return 0;
}
void movepla(int a[][W])//�л��ƶ�
{
	int i,j,dir;
    for(i=h-1;i>=0;i--)
    {
	    for(j=0;j<w;j++)
           {
				if((j==0||j==w-1)&&a[i][j]==4)//�л������߽�
             	     a[i][j]=0;
           		if(a[i][j]==4)
             	{
				 	a[i][j]=0;
             		dir=rand()%3;
				 	if(dir==0) //�����ƶ�
			 		  {
			 		    if(((i+1)==plh&&(j==plw||j==(plw-1)||j==(plw+1)))||(i==plh&&(j==(plw-2)||j==(plw+2))))
                        {
                            a[i][j]=0;
                            life--;
                        }
                        else
                            a[i+1][j]=4;
					  }
			 		if(dir==1)//�����ƶ�
			 		  {
			 		      if((i==plh&&j==(plw-2))||((i==plh+1)&&(j==(plw-3))))
                          {
                              a[i][j]=0;
                              life--;
                          }
                          else
                            a[i][j+1]=4;
					  }
			 		if(dir==2)//�����ƶ�
			 		   {
			 		       if((i==plh&&j==(plw+2))||((i==plh+1)&&(j==(plw+3))))
                           {
                               a[i][j]=0;
                               life--;
                           }
                           else
                            a[i][j-1]=4;
					   }
				}

        	}
	}
}
void setCurPos(int i,int j)
{
    hOutput=GetStdHandle(STD_OUTPUT_HANDLE);
    COORD coord={i,j};
    CONSOLE_CURSOR_INFO cursor_info={1,0};
    SetConsoleCursorPosition(hOutput,coord);
    SetConsoleCursorInfo(hOutput,&cursor_info);
    CONSOLE_SCREEN_BUFFER_INFO buffer;
	GetConsoleScreenBufferInfo(hOutput, &buffer);
}
void draw(int i,int j)
{
    if(score<300){
        setCurPos(i-2,j);
        printf(" ");
        setCurPos(i+2,j);
        printf(" ");
        setCurPos(i+1,j);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
        printf("\\");
        setCurPos(i,j);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
        printf("*");
        setCurPos(i-1,j);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
        printf("/");
        setCurPos(i+3,j+1);
        printf(" ");
        setCurPos(i-3,j+1);
        printf(" ");
        setCurPos(i-2,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
        printf("/");
        setCurPos(i-1,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
        printf("*");
        setCurPos(i,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
        printf("*");
        setCurPos(i+1,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
        printf("*");
        setCurPos(i+2,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
        printf("\\");
        setCurPos(i-2,j+2);
        printf("     ");
        setCurPos(i-2,j-1);
        printf("     ");
    }
    else if(score>=300){
        setCurPos(i-2,j);
        printf(" ");
        setCurPos(i-3,j);
        printf(" ");
        setCurPos(i+2,j);
        printf(" ");
        setCurPos(i+3,j);
        printf(" ");
        setCurPos(i+1,j);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
        printf("\\");
        setCurPos(i,j);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
        printf("*");
        setCurPos(i-1,j);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
        printf("/");
        setCurPos(i+4,j+1);
        printf(" ");
        setCurPos(i-4,j+1);
        printf(" ");
        setCurPos(i-3,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
        printf("/");
        setCurPos(i-2,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
        printf("+");
        setCurPos(i-1,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
        printf("*");
        setCurPos(i,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
        printf("*");
        setCurPos(i+1,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
        printf("*");
        setCurPos(i+2,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
        printf("+");
        setCurPos(i+3,j+1);
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
        printf("\\");
        setCurPos(i-3,j+2);
        printf("        ");
        setCurPos(i-3,j-1);
        printf("        ");
    }
}
void write()
{
    int i,j;
    static x=-1,y=-1;
    //int ch[100][100]={0};
    for(i=0;i<h;i++)
        for(j=0;j<w-1;j++)
    {
        if(i==0||i==h-1) scr[i][j]=5;
        else if((i!=0||i!=h-1)&&(j==0||j==w-2))
            scr[i][j]=5;
    }
    for(i=0;i<h;i++)
    {
        for(j=0;j<w;j++)
        if(j==w-1||change(i,j))
        {
            setCurPos(j,i);
            if(i==0&&j==w-1){
                SetConsoleTextAttribute(hOutput, FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED);
                printf(" Score:%d",score);
            }
            if(i==1&&j==w-1){
                SetConsoleTextAttribute(hOutput, FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED);
                printf(" Life:%d",life);
            }
            if(i==2&&j==w-1){
                SetConsoleTextAttribute(hOutput, FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED);
                printf(" Press 'S' escape game");
            }
            if (scr[i][j] == 5)
			{
				SetConsoleTextAttribute(hOutput,  BACKGROUND_BLUE |FOREGROUND_INTENSITY |FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_RED);
				printf("+");
			}
            if (scr[i][j] == 0)
			{
				printf(" ");
			}
			if (scr[i][j] == 2)
			{
                x=j;
                y=i;
			}
			if (scr[i][j] == 4)
			{
			    SetConsoleTextAttribute(hOutput, FOREGROUND_INTENSITY | FOREGROUND_RED);
				printf("*");
                //ch[i][j]=3;
			}
			if (scr[i][j] == 1)
			{
				SetConsoleTextAttribute(hOutput, FOREGROUND_INTENSITY | FOREGROUND_GREEN | FOREGROUND_RED  );
				printf("|");
			}

        }

    }
    //endraw(ch);
    draw(x,y);
}
void newpla() //�л�����
{
	srand(time(NULL));
    scr[1][rand()%w]=4;

}
void getrank()
{
	int k,i,temp,counta;
	char c;
	FILE *fp;
	fp=fopen(path,"a+");
	system("cls");
	for(i=0;i<30;i++)//���ϱ߽�
        {
            setCurPos(i,0);
            SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            printf("*");
        }
        printf("\n");
	setCurPos(0,1);
    SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ��
	rewind(fp);
	counta=fread(&c,sizeof(char),1,fp);
	if(counta==0)

		printf("No Record\n");
	else
	{
		rewind(fp);
		fscanf(fp,"%d",&k);
		for(i=1;i<=k;i++)
		{
			fscanf(fp,"%d",&temp);
			printf("No.%d %6d\n",i,temp);
		}
	}
	fclose(fp);
	printf("any key->back to menu");
	for(i=0;i<30;i++)//���ϱ߽�
        {
            setCurPos(i,k+2);
            SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            printf("*");
        }
	getchar();
}
void ranka()
{
	int i=1,k,ranking=0,temp,lista[101],counta;
	char c;
	//k���ļ���¼����ranking�����ֳɼ�������list�����гɼ���������
	FILE *fp;//�����ļ�ָ��
	fp=fopen(path,"a+");//����Ϊlog.txt���ļ�
	rewind(fp);
	counta=fread(&c,sizeof(char),1,fp);
	if(counta==0)
	{
		lista[1]=score;
		ranking=1;
		k=0;
	}
	else
	{
		rewind(fp);


		fscanf(fp,"%d",&k);
		for(i=1;i<=k;i++)
		{
			fscanf(fp,"%d",&temp);
			if(temp>=score)
				lista[i]=temp;
			else if (ranking==0)
			{
				lista[i]=score;
				lista
				[i+1]=temp;
				ranking=i;
			}
			else
				lista[i+1]=temp;
		}
		if (ranking==0)
		{
			ranking=k+1;
			lista[k+1]=score;
		}
	}
	fclose(fp);
	fp=fopen(path,"w+");
	fprintf(fp,"%d\n",k+1);//����ɼ���
	for(i=1;i<=k+1;i++)
		fprintf(fp,"%d\n",lista[i]);//��������ɼ�
	fclose(fp);//�ر��ļ�
	system("cls");//׼��output
	for(i=0;i<30;i++)//���ϱ߽�
        {
            setCurPos(i,0);
            SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
            printf("*");
        }
    printf("\n");
	setCurPos(0,1);
    SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ��
	printf("GOOD GAME!\nScore=%d Ranking=%d\n",score,ranking);//����þֳɼ�+����
	for(i=1;(i<=(k+1))&& (i<=10);i++)
		printf("No.%d %6d\n",i,lista[i]);//�ɼ���¼��>10,���ǰ10�� ;����������гɼ���¼

	printf("any key->back to menu\n");
	SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
	printf("******************************\n");//���һ�еı߿�
	getchar();
	getchar();
}
void settings(int flag)//flag==1:��Ϸ��0��δ��ʼ��Ϸ
{
	char c;
	int i,temp,sizea[2][4]={{0,25,30,40},{0,30,40,60}};
	while(1)
	{
	system("cls");

	printf("\n");
	for(i=0;i<20;i++)//���ϱ߽�
        {
            setCurPos(i,0);
            SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            printf("*");
        }
    printf("\n");
    SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
	printf("  1.speed   %d\n",speed);
	printf("  2.density %d\n",density);
	printf("  3.size    %dx%d\n",h,w);
	for(i=0;i<20;i++)//���ϱ߽�
        {
            setCurPos(i,4);
            SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            printf("*");
        }
        printf("\n");

	if (flag)
		printf("any other key->BACK TO GAME\n");
	else
		printf("any other key->MENU\n");
    for(i=0;i<30;i++)//�ָ���
        {
            setCurPos(i,6);
            SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
            printf("-");
        }
        printf("\n");
 	c=getchar();
 	for(i=0;i<30;i++)//���ϱ߽�
        {
            setCurPos(i,8);
            SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            printf("*");
        }
        printf("\n");
        for(i=0;i<30;i++)//���±߽�
    {
        setCurPos(i,11);
        SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
        printf("*");
    }
    setCurPos(0,9);//�ƻع��
    SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//������Ϊ��ɫ
 	if(c=='1')
	{
		printf("Write the new speed(50-200)\n");
	 	scanf("%d",&temp);
	 	if(temp>=50&& temp<=200)
	 		speed=temp;
	 	else
	 		printf("Error\n");
	}
	else if(c=='2')
	{
		printf("Write the new density(100-500)\n");
	 	scanf("%d",&temp);
	 	if(temp>=100 && temp<=500)
	 		density=temp;
	 	else
	 		printf("Error\n");
	}
	else if(c=='3')
	{
		if (flag)
			{
				printf("You are in the game,now can't change the size ");
				getchar();
			}
		else
		{
			printf("1.%dx%d 2.%dx%d 3.%dx%d\n",sizea[0][1],sizea[1][1],
					sizea[0][2],sizea[1][2],sizea[0][3],sizea[1][3]);
			scanf("%d",&temp);
			if(temp>=1 && temp<=3)
			{
				h=sizea[0][temp];
				w=sizea[1][temp];
			}
			else
				printf("Error\n");
		}
	}
	else
		break;
	getchar();
	}
	system("cls");
}
int read()
{
	int i,j;
	if(kbhit())
  	{
		switch(getch())
      	{
      		case 75://left
      		    if(score<300){
       			if(plw>4)
      			{
      			    if(scr[plh][plw-2]==4||scr[plh+1][plw-3]==4)
                    {
                        scr[plh][plw-2]=scr[plh+1][plw-3]=0;
                        life--;
                    }
      				scr[plh][plw]=0;plw--;scr[plh][plw]=2;
				}
      		    }
      		    else if(score>=300){
                   if(plw>6)
      			{
      			    if(scr[plh][plw-2]==4||scr[plh+1][plw-4]==4)
                    {
                        scr[plh][plw-2]=scr[plh+1][plw-4]=0;
                        life--;
                    }
      				scr[plh][plw]=0;plw--;scr[plh][plw]=2;
				}
      		    }
   				break;
          	case 77://right
          	    if(score<300){
       			if(plw<w-6)
       			{
       			    if(scr[plh][plw+2]==4||scr[plh+1][plw+3]==4)
                    {
                        scr[plh][plw+2]=scr[plh+1][plw+3]=0;
                        life--;
                    }
      				scr[plh][plw]=0;plw++;scr[plh][plw]=2;
				}
          	    }
          	    else if(score>=300){
                    if(plw<w-8)
       			{
       			    if(scr[plh][plw+2]==4||scr[plh+1][plw+4]==4)
                    {
                        scr[plh][plw+2]=scr[plh+1][plw+4]=0;
                        life--;
                    }
      				scr[plh][plw]=0;plw++;scr[plh][plw]=2;
				}
          	    }
               	break;
          	case 72://up
          	    if(score<300){
          		if(plh>2)
          		{
          		    if(scr[plh-1][plw]==4||scr[plh-1][plw-1]==4||scr[plh-1][plw+1]==4||scr[plh][plw-2]==4||scr[plh][plw+2]==4)
                    {
                        scr[plh-1][plw]=scr[plh-1][plw-1]=scr[plh-1][plw+1]=scr[plh][plw-2]=scr[plh][plw+2]=0;
                        life--;
                    }
       				scr[plh][plw]=0;plh--;scr[plh][plw]=2;
				}
          	    }
          	    else if(score>=300){
                    if(plh>2)
          		{
          		    if(scr[plh-1][plw]==4||scr[plh-1][plw-1]==4||scr[plh-1][plw+1]==4||scr[plh][plw-2]==4||scr[plh][plw+2]==4||scr[plh][plw-3]==4||scr[plh][plw+3]==4)
                    {
                        scr[plh-1][plw]=scr[plh-1][plw-1]=scr[plh-1][plw+1]=scr[plh][plw-2]=scr[plh][plw+2]=scr[plh][plw-3]=scr[plh][plw+3]=0;
                        life--;
                    }
       				scr[plh][plw]=0;plh--;scr[plh][plw]=2;
				}
          	    }
       			break;
       		case 80://down
       		    if(score<300){
       			if(plh<h-4)
       			{
       			    if(scr[plh+2][plw]==4||scr[plh+2][plw-1]==4||scr[plh+2][plw+1]==4||scr[plh+2][plw-2]==4||scr[plh+2][plw+2]==4)
                    {
                        scr[plh+2][plw]=scr[plh+2][plw-1]=scr[plh][plw+1]=scr[plh+2][plw-2]=scr[plh+2][plw+2]=0;
                        life--;
                    }
       			    scr[plh][plw]=0;plh++;scr[plh][plw]=2;
				}
       		    }
       		    else if(score>=300){
                if(plh<h-4)
       			{
       			    if(scr[plh+2][plw]==4||scr[plh+2][plw-1]==4||scr[plh+2][plw+1]==4||scr[plh+2][plw-2]==4||scr[plh+2][plw+2]==4||scr[plh+2][plw-3]==4||scr[plh+2][plw+3]==4)
                    {
                        scr[plh+2][plw]=scr[plh+2][plw-1]=scr[plh][plw+1]=scr[plh+2][plw-2]=scr[plh+2][plw+2]=scr[plh+2][plw-3]=scr[plh+2][plw+3]=0;
                        life--;
                    }
       			    scr[plh][plw]=0;plh++;scr[plh][plw]=2;
				}
       		    }
       			break;
  			case 83:  case 115://S����setting
      			settings(1);
      			for(i=0;i<=101;i++)
					for(j=0;j<=101;j++)
						lastscr[i][j]=0;
              	break;
            case 27:
            	return 0;
            	break;
            case ' ':
            	scr[plh-1][plw]=1;
            	if(score>=100)
                    scr[plh-1][plw-1]=scr[plh-1][plw+1]=1;
                if(score>=200)
                    scr[plh-1][plw-2]=scr[plh-1][plw+2]=1;
                if(score>=400)
                    scr[plh-1][plw-3]=scr[plh-1][plw+3]=1;
            	break;
        }
    }
    	return 1;
}
void rules(void)
{
    int i;
    printf("\n\n");
    for(i=0;i<105;i++)//���ϱ߽�
        {
            setCurPos(i,12);
            SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            printf("*");
        }
    printf("\n");
    SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);
    printf("Announce:\n");
    printf("Press the direction key to control your plane\n");
    printf("Press space to fire\n");
    printf("If you destory an enemy ,you'll get 10 score\n");
    printf("Every 100 score ,your plane will level up until you get 400 score,and your life piont will increase 1\n");
    printf("In the beginning, you have 3 life pionts. If your plane crash enemy your life pionts will decrease 1\n");
    for(i=0;i<105;i++)//���±߽�
        {
            setCurPos(i,19);
            SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
            printf("*");
        }
	getchar();
}
int menu()
{
	char c,d;
	int i;
	while(1)
	{
		system("cls");
		for(i=0;i<46;i++)//���ϱ߽�
        {
            setCurPos(i,0);
            SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
            printf("*");
        }
        printf("\n");
        SetConsoleTextAttribute(hOutput, FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);//��ɫ
		printf("                 START GAME\n");
		printf("                     /\\\n");
		printf("                     /\\\n");
		printf("                     /\\\n");
		printf("    SETTINGS  <<<<<<    >>>>>>  RANK\n ");
		printf("                    \\/\n");
		printf("                     \\/\n");
		printf("                     \\/\n");
		printf("                    RULES\n");
		for(i=0;i<46;i++)//���±߽�
        {
            setCurPos(i,10);
            SetConsoleTextAttribute(hOutput,FOREGROUND_GREEN | FOREGROUND_BLUE | FOREGROUND_INTENSITY);//��ɫ
            printf("*");
        }
		d=getch();
		c=getch();
		if(d==27)
            return 0;
		if(d==-32)
		{
			if(c==72)
				return 1;//�ص�main��ʼ��Ϸ
			else if(c==75)
			 	settings(0);//��ת������
			else if(c==77)
				getrank();//��ת��ȫ����
            else if(c==80)
                rules();//����0����main�б�ʾ�˳�����
		}
        else
            return 0;
		getchar();
	}

}
void movebul(int a[][W])
{
    int i,j;
    for(i=1;i<h;i++)
    {
        for(j=0;j<w;j++)
        {
            if(i==1&&a[i][j]==1)
                a[i][j]=0;
            if(a[i][j]==1)
            {
                if(a[i-1][j]==4)
                {
                    a[i][j]=a[i-1][j]=0;
                    score+=10;
                    if(score%100==0)
                        life++;
                }
                else
                {
                   	a[i][j]=0;
					a[i-1][j]=1;
				}
            }

        }
    }
}

int main()
{
    setCurPos(0,0);
    SetConsoleTextAttribute(hOutput,FOREGROUND_INTENSITY|FOREGROUND_GREEN|FOREGROUND_RED);
    printf("                                     WARNNING!!!  WARNNING!!!                        \n");
    printf("                 <<<<Warnning:Before start game,Please into full screen mode!!!!>>>>\n");
    printf("\n");
    printf("                 <<<<Warnning:Before start game,Please into full screen mode!!!!>>>>\n");
    getchar();
	int i,j;

	while(1)
	{
		if (!menu())//�˴����޸ģ�menu��������123֮���ֵ�򷵻�0����ʾ�˳�����
			break;
		//rules();
		system("cls");
		for(i=0;i<=101;i++)
			for(j=0;j<=101;j++)
				scr[i][j]=lastscr[i][j]=0;
        plh=h-4;
        plw=w/2;
        scr[plh][plw]=2;
        life=3;
        t=0;
        score=0;
        newpla();
        do
		{
			if (!read())//�˴����޸ģ�read��������esc�򷵻�0����ʾ�˳��þ���Ϸ������rank
				break;
			if (t%density==0)
				newpla();
			if (t%speed==0)
				movepla(scr);
            if(t%16==0)
                movebul(scr);
			write();
	//		Sleep(10);
			t++;
			if(t==30000)
                t=0;
		}
		while(life>0);
		ranka();
	}
	return 0;
}

