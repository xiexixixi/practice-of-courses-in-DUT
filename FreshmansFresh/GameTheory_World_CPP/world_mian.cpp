#include <iostream>
#include <cstdlib>
#include <string>
#include <windows.h>
#include <time.h>
#include <fstream>
#include <cmath>

/*每增加新的类型，需要
改const int TOTLESTRA 的值；
增加static int num 和void showNum（）,getNum()；
修改构造函数和析构函数
增加随机生成策略；
showNum();
增加switch（stra）；
删除ID；
*/
using namespace std;
const double NOISE = 0.1;  //噪音大小
const int MAXTURN = 400;   //个体间博弈最大轮数
const int k = 0.1;		   //Fermi策略修改项，k<=1，k越大理性程度越低
const float t = 4;		   //Moran策略优化项，t<=8, t越大理性程度越低
const int GEN = 400;	   //整体博弈轮数
const int WORLDSIZE = 12;  //模拟世界边界大小
const int TOTLESTRA = 5;   //策略总数	
static int countdown = 25; //判断结束

const int AVE = 1;
const int RAND = 0;
float grade[WORLDSIZE][WORLDSIZE] = { 0 };
short ChangeToStra[WORLDSIZE][WORLDSIZE]{ 0 };

void coloration(char c)//修改控制台颜色函数
{
	switch (c)
	{
	case 'b':case 'B':
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_BLUE);
		break;
	case 'g':case 'G':
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN);
		break;
	case 'r':case 'R':
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_RED);
		break;
	case 'c':case 'C':
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);
		break;
	case 'p':case 'P':
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_BLUE | FOREGROUND_RED | FOREGROUND_INTENSITY);
		break;
	case 'y':case 'Y':
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_INTENSITY);
		break;
	case 'w':case 'W':
		SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), FOREGROUND_GREEN | FOREGROUND_RED | FOREGROUND_BLUE | FOREGROUND_INTENSITY);
		break;
	default:break;
	}
}

class PDRule
{
private:
	const float R;  //同时合作的奖励R
	const float S;  //被背叛的损失S
	float T;  //背叛的收益T
	float P;  //同时背叛的惩罚P
public:
	PDRule(float r = 0.5) :R(1), S(0), T(1 + r), P(r) {
		if (r <= 0 || r >= 1)
		{
			cout << "value of r is invalid ,set r to 0.5." << endl;
			r = 0.5;
			T = 1.5;
		}
	}
	float calculatePayoff(int stra1, int stra2);//根据两个参与博弈者的策略，计算第一个博弈者的收益
};
float PDRule::calculatePayoff(int stra1, int stra2)
{  //合作策略为1，背叛策略为0
	if (stra1 == 1 && stra2 == 1) return R;
	else if (stra1 == 1 && stra2 == 0) return S;
	else if (stra1 == 0 && stra2 == 1) return T;
	else return P;
}



class Player
{
protected:
	int oppStra;              // 对手上一轮策略
	short stra;
	const int x;
	const int y;
public:
	Player(int x, int y, int stra) :x(x), y(y), stra(stra)
	{
		oppStra = -1;           // 初始没有对手策略，设置为-1 
	}
	virtual ~Player() {}

	virtual int nextStra();   // 本轮采用的策略,该函数需要被子类重写 
	virtual void setOppStra(int oppS) { oppStra = oppS; }// 设置对手上一轮策略
	virtual void reset() { oppStra = -1; } // 恢复到初始设置
	virtual short getStra() { return stra; }

	int getX() { return x; }
	int getY() { return y; }
	//	virtual void setXY(int x_, int y_) { this->x = x_, this->y = y_; }
};
int Player::nextStra()
{
	return 1;   // 总是采用合作策略 ；
}


class Player01 : public Player    //总是背叛
{
private:
	static int num;
public:
	Player01(int x, int y) :Player(x, y, 1) { num++; }
	~Player01() { num--; }
	int nextStra() { return 0; }               // 重写父类该函数 
	static void showNum()
	{
		if (Player01::num == 0)
			cout << "there is no strategy 01 " << endl;
		else
			cout << "the number of strategy 01 is :" << Player01::num << endl;
	}
	static int getNum() { return num; }
};
class Player02 :public Player		//702
{
	/*
	201792074
	前二十局策略固定： 1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1
	并记录对手的前二十局的反应：
	1.对手全出0 ，对手是个坏人， 防他，后面全出  0
	2.对手全出1， 对手是个好人， 坑他，后面全出  0
	3.对手策略为   1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1， 那他八成是TFT或者TFT改进版了， 和他合作以获取最大收益，后面全出 1
	用一个二维数组 a[3][20] 来储存对手的这三种模式
	用一个数组   opp[20]    来储存对手的实际策略
	由于存在系统噪音， 设 S[i] = sum( (a[i][j]-opp[j])^2 )  (0<=j<20)
	采取S[i]最小的那个策略， 若相等则取编号大的策略

	如果能提前和全班的人打一遍，就可以预存出更多的模式以应对各种奇葩对手，但这算作弊了吧 orz。。。。

	*/

private:
	int a[3][20];   // 对手的三个模式 
	int init[20];	// 初始的那20个固定策略 
	int opp[20];
	int maxTurn;    // 记录maxTurn的，用不上 orz。。。。 
	int cnt;
	int diff1, diff2, diff3;  // 对应上文的S[1],s[2],s[3] 

	static int num;
public:
	Player02(int x, int y) :Player(x, y, 2)
	{
		int temp1[3][20] = { 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
			1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,
			1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1 };
		int temp2[20] = { 1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1 };
		for (int i = 0; i < 3; i++) {
			for (int j = 0; j < 20; j++) {
				a[i][j] = temp1[i][j];
			}
		}

		for (int i = 0; i < 20; i++) init[i] = temp2[i];
		maxTurn = cnt = diff1 = diff2 = diff3 = 0;
		num++;
	}
	~Player02() {
		num--;
	}


	virtual int nextStra() { 	//only God knows, orz......
		if (cnt < 20) return init[cnt];
		if (diff1 == 0 && diff2 == 0 && diff3 == 0) {
			for (int i = 0; i < 20; i++) diff1 += (a[0][i] - opp[i])*(a[0][i] - opp[i]);
			for (int i = 0; i < 20; i++) diff2 += (a[1][i] - opp[i])*(a[1][i] - opp[i]);
			for (int i = 0; i < 20; i++) diff3 += (a[2][i] - opp[i])*(a[2][i] - opp[i]);
		}
		int mind = diff3;
		if (mind > diff1) mind = diff1;
		if (mind > diff2) mind = diff2;
		if (mind == diff3) {
			return 1;
		}
		else {
			return 0;
		}
	}

	virtual void setOppStra(int oppS) {
		oppStra = oppS;
		if (cnt < 20) opp[cnt] = oppS;
		cnt++;
	}
	static void showNum()
	{
		if (Player02::num == 0)
			cout << "there is no strategy 02 " << endl;
		else
			cout << "the number of strategy 02 is :" << Player02::num << endl;
	}
	virtual void reset() {
		oppStra = -1;
		if (maxTurn == 0) {
			maxTurn = cnt;
		}
		cnt = 0;
		diff1 = diff2 = diff3 = 0;
	}
	static int getNum() { return num; }

};
class Player03 :public Player		//TFT
{
	static int num;
public:
	Player03(int x, int y) :Player(x, y, 3) { num++; }
	~Player03()
	{
		num--;
	}
	int nextStra()
	{
		if (oppStra == -1) return 1;
		else return oppStra;
	}// 重写父类该函数 
	static void showNum()
	{
		if (Player03::num == 0)
			cout << "there is no strategy 03 " << endl;
		else
			cout << "the number of strategy 03 is :" << Player03::num << endl;
	}
	static int getNum() { return num; }

};
class Player04 :public Player	//随机TFT
{
	static int num;
public:
	Player04(int x, int y) :Player(x, y, 4)
	{
		num++;
		k1 = 0.2;
		gailv = rand() / double(RAND_MAX);
	}
	~Player04()
	{
		num--;
	}
	float k1;
	float gailv;
	int nextStra();
	static void showNum()
	{
		if (Player04::num == 0)
			cout << "there is no strategy 04 " << endl;
		else
			cout << "the number of strategy 04 is :" << Player04::num << endl;
	}
	static int getNum() { return num; }

};
int Player04::nextStra()
{
	if (oppStra == -1) return 1;
	srand(time(0));
	if (oppStra == 0)
	{
		if (gailv < k1)//如果对方背叛，0.2概率继续合作
			return 1;
	}
	else return oppStra;
}
class Player05 : public Player     //lmz 
{
	/*  学号：201792320
	思路：TFT为基础，原谅一次背叛，连续两次背叛开始背叛对方（K较大时胜过TFT）
	*/
private:
	int lastStra;
	static int num;
public:
	Player05(int x, int y) :Player(x, y, 5) { num++; }
	~Player05() { num--; }
	int nextStra();
	static void showNum() {
		if (Player05::num == 0)
			cout << "there is no strategy 05 " << endl;
		else
			cout << "the number of strategy 05 is :" << Player05::num << endl;
	}
	static int getNum() { return num; }

};
int Player05::nextStra()
{
	if (oppStra == -1)
	{
		lastStra = -1;
		return 1;
	}
	else if (oppStra == 0 && lastStra == 0)
		return 0;
	else
	{
		lastStra = oppStra;
		return 1;
	}
}

int Player01::num = 0;
int Player02::num = 0;
int Player03::num = 0;
int Player04::num = 0;
int Player05::num = 0;

class ImitationRule
{
public:

	void FermiRule();
	void BTO();
	void MoranRule(float k);

private:

};

void ImitationRule::FermiRule()
{
	srand((unsigned)time(NULL));
	int loca;
	double P, W;
	for (int i = 0; i < WORLDSIZE; i++)
	{
		for (int j = 0; j < WORLDSIZE; j++)
		{
			loca = rand() % 8 + 1;
			P = rand() / double(RAND_MAX);
			switch (loca)
			{
			case 1:
				W = 1 / (1 + exp((grade[i][j] - grade[(i + WORLDSIZE - 1) % WORLDSIZE][(j + WORLDSIZE - 1) % WORLDSIZE]) / k));
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 2:
				W = 1 / (1 + exp((grade[i][j] - grade[(i + WORLDSIZE - 1) % WORLDSIZE][j]) / k));
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;
			case 3:
				W = 1 / (1 + exp((grade[i][j] - grade[(i + WORLDSIZE - 1) % WORLDSIZE][(j + 1) % WORLDSIZE]) / k));
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 4:
				W = 1 / (1 + exp((grade[i][j] - grade[i][(j + WORLDSIZE - 1) % WORLDSIZE]) / k));
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 5:
				W = 1 / (1 + exp((grade[i][j] - grade[i][(j + 1) % WORLDSIZE]) / k));
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 6:
				W = 1 / (1 + exp((grade[i][j] - grade[(i + 1) % WORLDSIZE][(j + WORLDSIZE - 1) % WORLDSIZE]) / k));
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 7:
				W = 1 / (1 + exp((grade[i][j] - grade[(i + 1) % WORLDSIZE][j]) / k));
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 8:
				W = 1 / (1 + exp((grade[i][j] - grade[(i + 1) % WORLDSIZE][(j + 1) % WORLDSIZE]) / k));
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			default:
				break;
			}




		}
	}
}
void ImitationRule::BTO()
{
	for (int r = 0; r < WORLDSIZE; r++)
	{
		float max = 0;
		for (int c = 0; c < WORLDSIZE; c++)
		{
			max = grade[r][c];
			ChangeToStra[r][c] = 0;
			if (max < grade[(r + WORLDSIZE - 1) % WORLDSIZE][(c + WORLDSIZE - 1) % WORLDSIZE])
			{
				max = grade[(r + WORLDSIZE - 1) % WORLDSIZE][(c + WORLDSIZE - 1) % WORLDSIZE];
				ChangeToStra[r][c] = 1;
			}//左上

			if (max < grade[(r + WORLDSIZE - 1) % WORLDSIZE][c])
			{
				max = grade[(r + WORLDSIZE - 1) % WORLDSIZE][c];
				ChangeToStra[r][c] = 2;
			}//上

			if (max < grade[(r + WORLDSIZE - 1) % WORLDSIZE][(c + 1) % WORLDSIZE])
			{
				max = grade[(r + WORLDSIZE - 1) % WORLDSIZE][(c + 1) % WORLDSIZE];
				ChangeToStra[r][c] = 3;
			}//右上

			if (max < grade[r][(c + WORLDSIZE - 1) % WORLDSIZE])
			{
				max = grade[r][(c + WORLDSIZE - 1) % WORLDSIZE];
				ChangeToStra[r][c] = 4;
			}//左

			if (max < grade[r][(c + 1) % WORLDSIZE])
			{
				max = grade[r][(c + 1) % WORLDSIZE];
				ChangeToStra[r][c] = 5;
			}//右

			if (max < grade[(r + 1) % WORLDSIZE][(c + WORLDSIZE - 1) % WORLDSIZE])
			{
				max = grade[(r + 1) % WORLDSIZE][(c + WORLDSIZE - 1) % WORLDSIZE];
				ChangeToStra[r][c] = 6;
			}//左下

			if (max < grade[(r + 1) % WORLDSIZE][c])
			{
				max = grade[(r + 1) % WORLDSIZE][c];
				ChangeToStra[r][c] = 7;
			}//下

			if (max < grade[(r + 1) % WORLDSIZE][(c + 1) % WORLDSIZE])
			{
				max = grade[(r + 1) % WORLDSIZE][(c + 1) % WORLDSIZE];
				ChangeToStra[r][c] = 8;
			}//右下
		}
	}
}
void ImitationRule::MoranRule(float k_ = 4)
{
	int loca;
	float k = k_;
	float t = 4 / k;

	double P, W;
	double totlescore = 0;
	srand((unsigned)time(NULL));

	for (int i = 0; i < WORLDSIZE; i++)
	{
		for (int j = 0; j < WORLDSIZE; j++)
		{
			double totlescore = 0;
			for (int r = -1; r < 2; r++)
				for (int c = -1; c < 2; c++)
					if (c || r)
						totlescore += grade[(i + WORLDSIZE + r) % WORLDSIZE][(j + WORLDSIZE + c) % WORLDSIZE];
			//			cout << endl << totlescore << endl;

			loca = rand() % 8 + 1;
			P = rand() / double(RAND_MAX);
			switch (loca)
			{
			case 1:
				W = k * grade[(i + WORLDSIZE - 1) % WORLDSIZE][(j + WORLDSIZE - 1) % WORLDSIZE] / totlescore;
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 2:
				W = k * grade[(i + WORLDSIZE - 1) % WORLDSIZE][j] / totlescore;
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;
			case 3:
				W = k * grade[(i + WORLDSIZE - 1) % WORLDSIZE][(j + 1) % WORLDSIZE] / totlescore;
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 4:
				W = k * grade[i][(j + WORLDSIZE - 1) % WORLDSIZE] / totlescore;
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 5:
				W = k * grade[i][(j + 1) % WORLDSIZE] / totlescore;
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 6:
				W = k * grade[(i + 1) % WORLDSIZE][(j + WORLDSIZE - 1) % WORLDSIZE] / totlescore;
				if (P <= W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 7:
				W = k * grade[(i + 1) % WORLDSIZE][j] / totlescore;
				if (P <= t * W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			case 8:
				W = k * grade[(i + 1) % WORLDSIZE][(j + 1) % WORLDSIZE] / totlescore;
				if (P <= t * W)
					ChangeToStra[i][j] = loca;
				else
					ChangeToStra[i][j] = 0;
				break;

			default:
				break;
			}
		}
	}
	//	cout << endl << totlescore << endl;

}

class World
{
private:
	ImitationRule * imr;

public:
	Player * world[WORLDSIZE][WORLDSIZE];
	Player * preworld[WORLDSIZE][WORLDSIZE];

	void Record();
	World & createPlayers(int mode = 1);  //1是平均
	World & showGrade();
	World & show_Strategy();
	World & showStra();//show the strategies which players are going to changed to
	World & BTOStra();
	World & FermiStra();
	World & MoranStra(float kk);
	World & changeStra();
	void deletePlayer();
	void reset();
};
void World::Record()
{
	for (int j = 0; j < WORLDSIZE; j++)
	{
		for (int i = 0; i < WORLDSIZE; i++)
			if (preworld[i][j] != world[i][j])
			{
				preworld[i][j] = world[i][j];
			}
	}
}
World & World::createPlayers(int mode)
{
	int temp;
	int stranum[TOTLESTRA];
	if (!((WORLDSIZE * WORLDSIZE) % TOTLESTRA))
		for (int i = 0; i < TOTLESTRA; i++)
			stranum[i] = WORLDSIZE * WORLDSIZE / TOTLESTRA;
	else
	{
		for (int i = 0; i < TOTLESTRA; i++)
			stranum[i] = WORLDSIZE * WORLDSIZE / TOTLESTRA+1;
	}
	cout << stranum[0] << stranum[2] << stranum[4] << endl;
	srand((unsigned)time(0));
	if (mode == 1)
		for (int i = 0; i < WORLDSIZE; i++)
		{
			for (int j = 0; j < WORLDSIZE; j++)
			{
			start:
				temp = rand() % TOTLESTRA;
				switch (temp)
				{

				case 0:
					if ((stranum[0]--) > 0)
					{
						world[i][j] = new Player01(i, j);
						break;
					}
				case 1:
					if (stranum[1]-- > 0)
					{
						world[i][j] = new Player02(i, j);
						break;
					}
				case 2:
					if (stranum[2]-- > 0)
					{
						world[i][j] = new Player03(i, j);
						break;
					}
				case 3:
					if (stranum[3]-- > 0)
					{
						world[i][j] = new Player04(i, j);
						break;
					}
				case 4:
					if (stranum[4]-- > 0)
					{
						world[i][j] = new Player05(i, j);
						break;
					}
				default:
					goto start;
					break;
				}

			}
		}
	else if (mode == 0)
		for (int i = 0; i < WORLDSIZE; i++)
		{
			for (int j = 0; j < WORLDSIZE; j++)
			{
				temp = rand() % TOTLESTRA;

				if (temp == 0)
					world[i][j] = new Player01(i, j);
				else if (temp == 1)
					world[i][j] = new Player02(i, j);
				else if (temp == 2)
					world[i][j] = new Player03(i, j);
				else if (temp == 3)
					world[i][j] = new Player04(i, j);
				else if (temp == 4)
					world[i][j] = new Player05(i, j);
			}
		}

	return *this;
}
World & World::showGrade()
{
	for (int r = 0; r < WORLDSIZE; r++)
	{
		for (int c = 0; c < WORLDSIZE; c++)
			cout << grade[r][c] << "\t";
		cout << endl;
	}
	return *this;
}
World & World::show_Strategy()
{
	for (int i = 0; i < WORLDSIZE; i++)
	{
		for (int j = 0; j < WORLDSIZE; j++)
		{
			if (world[i][j]->getStra() == 1)
			{
				coloration('b');
				cout << world[i][j]->getStra() << "\t";

			}
			if (world[i][j]->getStra() == 2)
			{
				coloration('g');
				cout << world[i][j]->getStra() << "\t";

			}
			if (world[i][j]->getStra() == 3)
			{
				coloration('p');
				cout << world[i][j]->getStra() << "\t";

			}
			if (world[i][j]->getStra() == 4)
			{
				coloration('r');
				cout << world[i][j]->getStra() << "\t";

			}
			if (world[i][j]->getStra() == 5)
			{
				coloration('y');
				cout << world[i][j]->getStra() << "\t";

			}

		}
		//			cout << world[i][j]->getStra() << "\t";

		cout << endl;
	}
	return *this;
}
World & World::showStra()//show the strategies which players are going to changed to
{
	for (int r = 0; r < WORLDSIZE; r++)
	{
		for (int c = 0; c < WORLDSIZE; c++)
			cout << ChangeToStra[r][c] << "\t";
		cout << endl;
	}
	return *this;
}
World & World::BTOStra()//BTO   添加阶数
{
	imr->BTO();
	return *this;
}
World & World::FermiStra()
{
	imr->FermiRule();
	return *this;
}

World & World::MoranStra(float kk = 4)   //k->8时 选择随机性越大
{
	imr->MoranRule(kk);
	return *this;
	// TODO: 在此处插入 return 语句
}

void World::reset()
{
	for (int i = 0; i < WORLDSIZE; i++)
		for (int j = 0; j < WORLDSIZE; j++)
			grade[i][j] = 0;
}

World & World::changeStra()
{
	for (int r = 0; r < WORLDSIZE; r++)
	{
		for (int c = 0; c < WORLDSIZE; c++)
		{
			int stra;
			switch (ChangeToStra[r][c])
			{
			case 0:
				break;
			case 1://可写为函数
				delete world[r][c];
				stra = world[(r + WORLDSIZE - 1) % WORLDSIZE][(c + WORLDSIZE - 1) % WORLDSIZE]->getStra();
				switch (stra)
				{
				case 1:
					world[r][c] = new Player01(r, c);
					break;
				case 2:
					world[r][c] = new Player02(r, c);
					break;
				case 3:
					world[r][c] = new Player03(r, c);
					break;
				case 4:
					world[r][c] = new Player04(r, c);
					break;
				case 5:
					world[r][c] = new Player05(r, c);
					break;
				default:
					break;
				}

				break;
			case 2:
				delete world[r][c];
				stra = world[(r + WORLDSIZE - 1) % WORLDSIZE][c % WORLDSIZE]->getStra();
				switch (stra)
				{
				case 1:
					world[r][c] = new Player01(r, c);
					break;
				case 2:
					world[r][c] = new Player02(r, c);
					break;
				case 3:
					world[r][c] = new Player03(r, c);
					break;
				case 4:
					world[r][c] = new Player04(r, c);
					break;
				case 5:
					world[r][c] = new Player05(r, c);
					break;
				default:
					break;
				}
				break;

			case 3:
				delete world[r][c];
				stra = world[(r + WORLDSIZE - 1) % WORLDSIZE][(c + 1) % WORLDSIZE]->getStra();
				switch (stra)
				{
				case 1:
					world[r][c] = new Player01(r, c);
					break;
				case 2:
					world[r][c] = new Player02(r, c);
					break;
				case 3:
					world[r][c] = new Player03(r, c);
					break;
				case 4:
					world[r][c] = new Player04(r, c);
					break;
				case 5:
					world[r][c] = new Player05(r, c);
					break;
				default:
					break;
				}
				break;
			case 4:
				delete world[r][c];
				stra = world[r % WORLDSIZE][(c + WORLDSIZE - 1) % WORLDSIZE]->getStra();
				switch (stra)
				{
				case 1:
					world[r][c] = new Player01(r, c);
					break;
				case 2:
					world[r][c] = new Player02(r, c);
					break;
				case 3:
					world[r][c] = new Player03(r, c);
					break;
				case 4:
					world[r][c] = new Player04(r, c);
					break;
				case 5:
					world[r][c] = new Player05(r, c);
					break;
				default:
					break;
				}
				break;
			case 5:
				delete world[r][c];
				stra = world[r % WORLDSIZE][(c + 1) % WORLDSIZE]->getStra();
				switch (stra)
				{
				case 1:
					world[r][c] = new Player01(r, c);
					break;
				case 2:
					world[r][c] = new Player02(r, c);
					break;
				case 3:
					world[r][c] = new Player03(r, c);
					break;
				case 4:
					world[r][c] = new Player04(r, c);
					break;
				case 5:
					world[r][c] = new Player05(r, c);
					break;
				default:
					break;
				}
				break;
			case 6:
				delete world[r][c];
				stra = world[(r + 1) % WORLDSIZE][(c + WORLDSIZE - 1) % WORLDSIZE]->getStra();
				switch (stra)
				{
				case 1:
					world[r][c] = new Player01(r, c);
					break;
				case 2:
					world[r][c] = new Player02(r, c);
					break;
				case 3:
					world[r][c] = new Player03(r, c);
					break;
				case 4:
					world[r][c] = new Player04(r, c);
					break;
				case 5:
					world[r][c] = new Player05(r, c);
					break;
				default:
					break;
				}
				break;
			case 7:
				delete world[r][c];
				stra = world[(r + 1) % WORLDSIZE][c % WORLDSIZE]->getStra();
				switch (stra)
				{
				case 1:
					world[r][c] = new Player01(r, c);
					break;
				case 2:
					world[r][c] = new Player02(r, c);
					break;
				case 3:
					world[r][c] = new Player03(r, c);
					break;
				case 4:
					world[r][c] = new Player04(r, c);
					break;
				case 5:
					world[r][c] = new Player05(r, c);
					break;
				default:
					break;
				}
				break;

			case 8:
				delete world[r][c];
				stra = world[(r + 1) % WORLDSIZE][(c + 1) % WORLDSIZE]->getStra();
				switch (stra)
				{
				case 1:
					world[r][c] = new Player01(r, c);
					break;
				case 2:
					world[r][c] = new Player02(r, c);
					break;
				case 3:
					world[r][c] = new Player03(r, c);
					break;
				case 4:
					world[r][c] = new Player04(r, c);
					break;
				case 5:
					world[r][c] = new Player05(r, c);
					break;
				default:
					break;
				}
				break;

			default:
				break;
			}
		}
	}
	return *this;
}
void World::deletePlayer()
{
	for (int i = 0; i < WORLDSIZE; i++)
		for (int j = 0; j < WORLDSIZE; j++)
			delete world[i][j];// p[i][j];

}



class IPDG
{
private:
	const int maxTurn;  //game 进行的轮数 
	const float K;      //系统噪音 

	PDRule * pdr;

public:
	IPDG(int maxTurn, PDRule * pdr, float K) :
		maxTurn(maxTurn), pdr(pdr), K(K) {}
	void startGame(Player * p1, Player * p2);

};
void IPDG::startGame(Player * p1, Player * p2)
{
	/*（i1,就）为p1的坐标，（i2,j2）为p2的坐标.开始后不同方向两坐标相对位置不同*/
	p1->reset();
	p2->reset();/**/
	srand((unsigned)time(NULL));
	for (int i = 0; i < maxTurn; i++)
	{
		int s1 = p1->nextStra();
		int s2 = p2->nextStra();
		double rand1 = rand() / double(RAND_MAX);
		double rand2 = rand() / double(RAND_MAX);
		if (rand1 < K) s1 = 1 - s1;//player1受到噪音干扰
		if (rand2 < K) s2 = 1 - s2;//player2受到噪音干扰 
		grade[p1->getX()][p1->getY()] += pdr->calculatePayoff(s1, s2);
		grade[p2->getX()][p2->getY()] += pdr->calculatePayoff(s2, s1);
		p1->setOppStra(s2);
		p2->setOppStra(s1);
	}
}


class Result {     //存放博弈结果
	float payoffs;
public:
	Result() : payoffs(0) {}
	void setPayoffs(float p) { payoffs = p; }
	float getPayoffs() { return payoffs; }
	void inceasePayoffs(int p) { payoffs += p; }
};



int main()
{
	PDRule pdr(0.01);//   值越小越利于合作
	IPDG ipdg(MAXTURN, &pdr, NOISE);
	World newworld;
	int num = 0;

	/*生成策略*/
	newworld.createPlayers(1);   //参数值为：1 每种策略数量相等，0随机生成策略；
	newworld.Record();

	/*文件指针 打开Excel  （1.策略数量变化  2.策略分布图）*/
	ofstream outFile;
	outFile.open("show Strategy.csv", ios::out | ios::trunc);
	outFile << "Stra 1" << "," << "Stra 2" << "," << "Stra 3" << "," << "Stra 4" << "," << "Stra 5" << endl;
	outFile.close();

	/*开始博弈*/
	int  i = 0;
	while (i++ < GEN)
	{
		for (int r = 0; r < WORLDSIZE; r++)
			for (int c = 0; c < WORLDSIZE; c++)
				ipdg.startGame(newworld.world[r][c], newworld.world[r][(c + 1) % WORLDSIZE]);
		for (int c = 0; c < WORLDSIZE; c++)
			for (int r = 0; r < WORLDSIZE; r++)
				ipdg.startGame(newworld.world[r][c], newworld.world[(r + 1) % WORLDSIZE][c]);
		for (int r = 0; r < WORLDSIZE; r++)
			for (int c = 0; c < WORLDSIZE; c++)
				ipdg.startGame(newworld.world[r][c], newworld.world[(r + 1) % WORLDSIZE][(c + 1) % WORLDSIZE]);
		for (int r = 0; r < WORLDSIZE; r++)
			for (int c = 0; c < WORLDSIZE; c++)
				ipdg.startGame(newworld.world[r][c], newworld.world[(r + WORLDSIZE - 1) % WORLDSIZE][(c + 1) % WORLDSIZE]);
		/*计算邻居最大值,并定好策略*/

		/*每轮结束，记录数据：1.策略分布（颜色标记）；2.每种策略的数量*/
		coloration('r');
		newworld.showGrade().show_Strategy();
		coloration('b');
		Player01::showNum();
		Player02::showNum();
		Player03::showNum();
		Player04::showNum();
		Player05::showNum();
		cout << endl;

		/*存入数据*/

		outFile.open("show Strategy.csv", ios::out | ios::app);
		outFile << Player01::getNum() << "," << Player02::getNum() << "," << Player03::getNum() << ","
			<< Player04::getNum() << "," << Player05::getNum() << endl;
		outFile.close();
		/*分布图 文件*/
		outFile.open("Strategy distribution.csv", ios::out | ios::app);
		for (int i = 0; i < WORLDSIZE; i++)
		{
			for (int j = 0; j < WORLDSIZE; j++)
			{
				outFile << newworld.world[i][j]->getStra() << ",";
			}
			outFile << endl;
		}
		outFile << endl;
		outFile.close();
		/*改变策略*/
		newworld.FermiStra().changeStra();      //可更换策略

		for (int i = 0; i < WORLDSIZE; i++)
			for (int j = 0; j < WORLDSIZE; j++)
				grade[i][j] = 0;
		/*判断结果  自动结束  1.只剩一种策略  2.策略数量及位置保持countdown代不变*/
		for (int i = 0; i < WORLDSIZE; i++)
		{
			for (int j = 0; j < WORLDSIZE; j++)
			{
				if (newworld.preworld[i][j] != newworld.world[i][j])
					countdown = 10;
			}
		}
		countdown--;
		if (Player01::getNum() == WORLDSIZE * WORLDSIZE || Player02::getNum() == WORLDSIZE * WORLDSIZE ||
			Player03::getNum() == WORLDSIZE * WORLDSIZE || Player04::getNum() == WORLDSIZE * WORLDSIZE ||
			Player05::getNum() == WORLDSIZE * WORLDSIZE)
		{
			static int count_down = 3;
			if (!count_down)
				break;
			count_down--;
		}
		newworld.Record();

		cout << endl << countdown << endl;
		if (!countdown)
			break;
	}

	/*析构Players*/
	newworld.deletePlayer();
	system("pause");
}