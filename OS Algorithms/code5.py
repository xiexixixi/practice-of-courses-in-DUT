# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 23:06:00 2020

@author: Lenovo
"""
import random


class disk_block():
    def __init__(self,size = 2000):
        self.size = size
        self.is_empty = True
        
    def taken(self):
        self.is_empty = False
        
    def free_up(self):
        self.is_empty = True
        

class file():
    def __init__(self,name,size,content=None):
        self.name = name
        self.size = size
        self.address = None
        self.content = content
        
    def save(self,address):
        self.address = address
        
    def write(self,content):
        self.content = content
        
    def show_info(self):
        print("name:",self.name)
        print("address:",self.address)

        print("size:",self.size)
        print("content:",self.content)

    def info_text(self):
        info = "name:{}\naddress:{}\nsize:{}\ncontent:{}\n".format(
                self.name,self.address,self.size,self.content)
        return info

class disk_manager():
    def __init__(self,block_num = 500,block_size = 2000):
        self.block_num = block_num
        self.block_size = block_size
        self.vacancy_table = {1:[1,500]}
        self.files_num = 0
        
        self.free_blocks_num = block_num
        self.directory = {}
        
    def first_fit(self,file):
        for seq in self.vacancy_table:
            free_chunk = self.vacancy_table[seq]
            block_num = file.size//self.block_size+1
            if free_chunk[1]>=block_num:
                free_chunk = self.vacancy_table[seq].copy()
                start_block = free_chunk[0]+block_num
                update_free_block = free_chunk[1]-block_num
                del self.vacancy_table[seq]
                
                
                if update_free_block:
                    self.vacancy_table[seq] = [start_block,update_free_block]
                return free_chunk[0]
            
        else:
            
            return False
            
    
    def Create_file(self,name,size,content=None):
        f = file(name,size,content)
        if name in self.directory:
            print("There has been a file with same name.")
            return False
        
        else:
            f = file(name,size)
            
            blocks = size//self.block_size+1
            
            address = self.first_fit(f)
            
            if address:
                f.save(address)
                self.directory[name] = [address,blocks,f]
                print("Successfully created: address:%s blocks, size:%s blocks."%(address,blocks))
                self.files_num += 1
                self.free_blocks_num -= blocks
                return True
            else:
                print("Creation failed: insufficient storage space!")
                return False
                

    def Delete_file(self,name):
        
        if name not in self.directory:
            print("No such a file")
            return False
        
        address,size,_ = self.directory[name]

        next_seg_add = address+size
        
        pre_free = False
        next_free = False
        
        pre_free_seq = None
        next_free_seq = None
        #判断上下是否空闲
        
        for seq in self.vacancy_table:
            free_add,blocks = self.vacancy_table[seq]
            
            if free_add + blocks == address:
                pre_free = True
                pre_free_seq = seq
            
            if free_add == next_seg_add:
                next_free = True
                next_free_seq = seq
                
            
        #上下都是空闲表
        if pre_free and next_free:
            update_add = self.vacancy_table[pre_free_seq][0]
            update_blocks = self.vacancy_table[pre_free_seq][1]+self.vacancy_table[pre_free_seq][1]+size
            self.vacancy_table[pre_free_seq] = [update_add,update_blocks]
            
            del self.vacancy_table[next_free_seq]
        #上下都不是空闲表
        elif not (pre_free or next_free):
            seq
            min_key = min(self.vacancy_table.keys())
            max_key = max(self.vacancy_table.keys())
            for i in range(min_key+1,max_key+2):
                if i not in self.vacancy_table:
                    seq = i
            self.vacancy_table[seq] = [address,size]
            
        #上是空闲，下不空闲
        
        elif pre_free and not next_free:
            update_add = self.vacancy_table[pre_free_seq][0]
            update_blocks = self.vacancy_table[pre_free_seq][1]+size
            self.vacancy_table[pre_free_seq] = [update_add,update_blocks]
        #上不空闲，下空闲
        else:
            update_blocks = self.vacancy_table[next_free_seq][1]+size
            self.vacancy_table[next_free_seq] = [address,update_blocks]
        del self.directory[name]
        self.files_num-=1
        
        self.free_blocks_num += size
        print("Successfully deleted")
        return True

        
    
    def Search_file(self,name):
        file = self.directory.get(name,None)
        if file is None:
            print("No such a file")
            return None
        return file[2]
    
    def show_info(self):
        print("totle block num:%d"%self.block_num)
        print("block size:\t%dk"%(self.block_size//1000))
        print("files num:\t%d"%self.files_num)
        print("free blocks num:%d"%self.free_blocks_num)
        print("vacancy table:\t",self.vacancy_table)
        print("directory:")
        for key,value in self.directory.items():
            print(key,value[:2])
    def info_text(self):
        
        info = "totle block num:\t%d\nblock size:\t\t%dk\nfiles num:\t\t%d\nfree blocks num:\t%d"%(
                self.block_num,self.block_size//1000,self.files_num,self.free_blocks_num)

        return info
        
if __name__ == '__main__':
        
    dm = disk_manager()
    
    #随机生成50个文件
    for i in range(50):
        name = str(i)+'.txt'
        size = random.randint(2000,10000)
        dm.Create_file(name,size)
    
    
    for i in range(0,50,2):
        name = str(i)+'.txt'
        dm.Delete_file(name)
    
    dm.Create_file('A.txt',7000)
    dm.Create_file('B.txt',5000)
    dm.Create_file('C.txt',2000)
    dm.Create_file('D.txt',9000)
    dm.Create_file('E.txt',3500)
    
    dm.show_info()
    
    #dm.Delete_file(r'3.txt')
    #dm.show_info()




