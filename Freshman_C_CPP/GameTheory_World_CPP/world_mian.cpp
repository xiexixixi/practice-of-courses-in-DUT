#include <iostream>
#include <cstdlib>
#include <string>
#include <windows.h>
#include <time.h>
#include <fstream>
#include <cmath>

/*ÿ�����µ����ͣ���Ҫ
��const int TOTLESTRA ��ֵ��
����static int num ��void showNum����,getNum()��
�޸Ĺ��캯������������
����������ɲ��ԣ�
showNum();
����switch��stra����
ɾ��ID��
*/
using namespace std;
const double NOISE = 0.1;  //������С
const int MAXTURN = 400;   //����䲩���������
const int k = 0.1;		   //Fermi�����޸��k<=1��kԽ�����Գ̶�Խ��
const float t = 4;		   //Moran�����Ż��t<=8, tԽ�����Գ̶�Խ��
const int GEN = 400;	   //���岩������
const int WORLDSIZE = 12;  //ģ������߽��С
const int TOTLESTRA = 5;   //��������	
static int countdown = 25; //�жϽ���

const int AVE = 1;
const int RAND = 0;
float grade[WORLDSIZE][WORLDSIZE] = { 0 };
short ChangeToStra[WORLDSIZE][WORLDSIZE]{ 0 };

void coloration(char c)//�޸Ŀ���̨��ɫ����
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
	const float R;  //ͬʱ�����Ľ���R
	const float S;  //�����ѵ���ʧS
	float T;  //���ѵ�����T
	float P;  //ͬʱ���ѵĳͷ�P
public:
	PDRule(float r = 0.5) :R(1), S(0), T(1 + r), P(r) {
		if (r <= 0 || r >= 1)
		{
			cout << "value of r is invalid ,set r to 0.5." << endl;
			r = 0.5;
			T = 1.5;
		}
	}
	float calculatePayoff(int stra1, int stra2);//�����������벩���ߵĲ��ԣ������һ�������ߵ�����
};
float PDRule::calculatePayoff(int stra1, int stra2)
{  //��������Ϊ1�����Ѳ���Ϊ0
	if (stra1 == 1 && stra2 == 1) return R;
	else if (stra1 == 1 && stra2 == 0) return S;
	else if (stra1 == 0 && stra2 == 1) return T;
	else return P;
}



class Player
{
protected:
	int oppStra;              // ������һ�ֲ���
	short stra;
	const int x;
	const int y;
public:
	Player(int x, int y, int stra) :x(x), y(y), stra(stra)
	{
		oppStra = -1;           // ��ʼû�ж��ֲ��ԣ�����Ϊ-1 
	}
	virtual ~Player() {}

	virtual int nextStra();   // ���ֲ��õĲ���,�ú�����Ҫ��������д 
	virtual void setOppStra(int oppS) { oppStra = oppS; }// ���ö�����һ�ֲ���
	virtual void reset() { oppStra = -1; } // �ָ�����ʼ����
	virtual short getStra() { return stra; }

	int getX() { return x; }
	int getY() { return y; }
	//	virtual void setXY(int x_, int y_) { this->x = x_, this->y = y_; }
};
int Player::nextStra()
{
	return 1;   // ���ǲ��ú������� ��
}


class Player01 : public Player    //���Ǳ���
{
private:
	static int num;
public:
	Player01(int x, int y) :Player(x, y, 1) { num++; }
	~Player01() { num--; }
	int nextStra() { return 0; }               // ��д����ú��� 
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
	ǰ��ʮ�ֲ��Թ̶��� 1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1
	����¼���ֵ�ǰ��ʮ�ֵķ�Ӧ��
	1.����ȫ��0 �������Ǹ����ˣ� ����������ȫ��  0
	2.����ȫ��1�� �����Ǹ����ˣ� ����������ȫ��  0
	3.���ֲ���Ϊ   1,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1,1,0,1�� �����˳���TFT����TFT�Ľ����ˣ� ���������Ի�ȡ������棬����ȫ�� 1
	��һ����ά���� a[3][20] ��������ֵ�������ģʽ
	��һ������   opp[20]    ��������ֵ�ʵ�ʲ���
	���ڴ���ϵͳ������ �� S[i] = sum( (a[i][j]-opp[j])^2 )  (0<=j<20)
	��ȡS[i]��С���Ǹ����ԣ� �������ȡ��Ŵ�Ĳ���

	�������ǰ��ȫ����˴�һ�飬�Ϳ���Ԥ��������ģʽ��Ӧ�Ը���������֣������������˰� orz��������

	*/

private:
	int a[3][20];   // ���ֵ�����ģʽ 
	int init[20];	// ��ʼ����20���̶����� 
	int opp[20];
	int maxTurn;    // ��¼maxTurn�ģ��ò��� orz�������� 
	int cnt;
	int diff1, diff2, diff3;  // ��Ӧ���ĵ�S[1],s[2],s[3] 

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
	}// ��д����ú��� 
	static void showNum()
	{
		if (Player03::num == 0)
			cout << "there is no strategy 03 " << endl;
		else
			cout << "the number of strategy 03 is :" << Player03::num << endl;
	}
	static int getNum() { return num; }

};
class Player04 :public Player	//���TFT
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
		if (gailv < k1)//����Է����ѣ�0.2���ʼ�������
			return 1;
	}
	else return oppStra;
}
class Player05 : public Player     //lmz 
{
	/*  ѧ�ţ�201792320
	˼·��TFTΪ������ԭ��һ�α��ѣ��������α��ѿ�ʼ���ѶԷ���K�ϴ�ʱʤ��TFT��
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
			}//����

			if (max < grade[(r + WORLDSIZE - 1) % WORLDSIZE][c])
			{
				max = grade[(r + WORLDSIZE - 1) % WORLDSIZE][c];
				ChangeToStra[r][c] = 2;
			}//��

			if (max < grade[(r + WORLDSIZE - 1) % WORLDSIZE][(c + 1) % WORLDSIZE])
			{
				max = grade[(r + WORLDSIZE - 1) % WORLDSIZE][(c + 1) % WORLDSIZE];
				ChangeToStra[r][c] = 3;
			}//����

			if (max < grade[r][(c + WORLDSIZE - 1) % WORLDSIZE])
			{
				max = grade[r][(c + WORLDSIZE - 1) % WORLDSIZE];
				ChangeToStra[r][c] = 4;
			}//��

			if (max < grade[r][(c + 1) % WORLDSIZE])
			{
				max = grade[r][(c + 1) % WORLDSIZE];
				ChangeToStra[r][c] = 5;
			}//��

			if (max < grade[(r + 1) % WORLDSIZE][(c + WORLDSIZE - 1) % WORLDSIZE])
			{
				max = grade[(r + 1) % WORLDSIZE][(c + WORLDSIZE - 1) % WORLDSIZE];
				ChangeToStra[r][c] = 6;
			}//����

			if (max < grade[(r + 1) % WORLDSIZE][c])
			{
				max = grade[(r + 1) % WORLDSIZE][c];
				ChangeToStra[r][c] = 7;
			}//��

			if (max < grade[(r + 1) % WORLDSIZE][(c + 1) % WORLDSIZE])
			{
				max = grade[(r + 1) % WORLDSIZE][(c + 1) % WORLDSIZE];
				ChangeToStra[r][c] = 8;
			}//����
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
	World & createPlayers(int mode = 1);  //1��ƽ��
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
World & World::BTOStra()//BTO   ��ӽ���
{
	imr->BTO();
	return *this;
}
World & World::FermiStra()
{
	imr->FermiRule();
	return *this;
}

World & World::MoranStra(float kk = 4)   //k->8ʱ ѡ�������Խ��
{
	imr->MoranRule(kk);
	return *this;
	// TODO: �ڴ˴����� return ���
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
			case 1://��дΪ����
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
	const int maxTurn;  //game ���е����� 
	const float K;      //ϵͳ���� 

	PDRule * pdr;

public:
	IPDG(int maxTurn, PDRule * pdr, float K) :
		maxTurn(maxTurn), pdr(pdr), K(K) {}
	void startGame(Player * p1, Player * p2);

};
void IPDG::startGame(Player * p1, Player * p2)
{
	/*��i1,�ͣ�Ϊp1�����꣬��i2,j2��Ϊp2������.��ʼ��ͬ�������������λ�ò�ͬ*/
	p1->reset();
	p2->reset();/**/
	srand((unsigned)time(NULL));
	for (int i = 0; i < maxTurn; i++)
	{
		int s1 = p1->nextStra();
		int s2 = p2->nextStra();
		double rand1 = rand() / double(RAND_MAX);
		double rand2 = rand() / double(RAND_MAX);
		if (rand1 < K) s1 = 1 - s1;//player1�ܵ���������
		if (rand2 < K) s2 = 1 - s2;//player2�ܵ��������� 
		grade[p1->getX()][p1->getY()] += pdr->calculatePayoff(s1, s2);
		grade[p2->getX()][p2->getY()] += pdr->calculatePayoff(s2, s1);
		p1->setOppStra(s2);
		p2->setOppStra(s1);
	}
}


class Result {     //��Ų��Ľ��
	float payoffs;
public:
	Result() : payoffs(0) {}
	void setPayoffs(float p) { payoffs = p; }
	float getPayoffs() { return payoffs; }
	void inceasePayoffs(int p) { payoffs += p; }
};



int main()
{
	PDRule pdr(0.01);//   ֵԽСԽ���ں���
	IPDG ipdg(MAXTURN, &pdr, NOISE);
	World newworld;
	int num = 0;

	/*���ɲ���*/
	newworld.createPlayers(1);   //����ֵΪ��1 ÿ�ֲ���������ȣ�0������ɲ��ԣ�
	newworld.Record();

	/*�ļ�ָ�� ��Excel  ��1.���������仯  2.���Էֲ�ͼ��*/
	ofstream outFile;
	outFile.open("show Strategy.csv", ios::out | ios::trunc);
	outFile << "Stra 1" << "," << "Stra 2" << "," << "Stra 3" << "," << "Stra 4" << "," << "Stra 5" << endl;
	outFile.close();

	/*��ʼ����*/
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
		/*�����ھ����ֵ,�����ò���*/

		/*ÿ�ֽ�������¼���ݣ�1.���Էֲ�����ɫ��ǣ���2.ÿ�ֲ��Ե�����*/
		coloration('r');
		newworld.showGrade().show_Strategy();
		coloration('b');
		Player01::showNum();
		Player02::showNum();
		Player03::showNum();
		Player04::showNum();
		Player05::showNum();
		cout << endl;

		/*��������*/

		outFile.open("show Strategy.csv", ios::out | ios::app);
		outFile << Player01::getNum() << "," << Player02::getNum() << "," << Player03::getNum() << ","
			<< Player04::getNum() << "," << Player05::getNum() << endl;
		outFile.close();
		/*�ֲ�ͼ �ļ�*/
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
		/*�ı����*/
		newworld.FermiStra().changeStra();      //�ɸ�������

		for (int i = 0; i < WORLDSIZE; i++)
			for (int j = 0; j < WORLDSIZE; j++)
				grade[i][j] = 0;
		/*�жϽ��  �Զ�����  1.ֻʣһ�ֲ���  2.����������λ�ñ���countdown������*/
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

	/*����Players*/
	newworld.deletePlayer();
	system("pause");
}