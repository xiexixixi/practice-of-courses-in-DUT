#pragma once
#include<stdlib.h>
#include<stdio.h>
#include<windows.h>
#include<math.h>
#include<string.h>
#include"sort.h"
#include"translate.h"
void write(struct map *arr, FILE* fp);/*������д���ļ�����*/
void linkid_binary_search(struct map *arr[], int length);/*linkid���������ַ�������*/
void roadname_search(struct map *arr[], int length);/*·������*/
void classnumber_search(struct map *arr[], int length);/*Class���ż���*/
void sideway_search(struct map *arr[], int length);/*��·�ڼ���*/