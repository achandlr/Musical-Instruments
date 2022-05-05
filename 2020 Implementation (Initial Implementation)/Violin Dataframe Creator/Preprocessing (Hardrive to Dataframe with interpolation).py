#!/usr/bin/env python
# coding: utf-8

# In[ ]:


DESIRED_COUNTRIES=7

#!/usr/bin/env python
# coding: utf-8

# # Process Data to DataFrame and Array images

# In[2]:
import gc
gc.enable()
SIZE_IMAGE=128
import glob, os, fnmatch
import json
import numpy
from numpy import save
from PIL import Image
from numpy import asarray
# from InflationEstimator import InflationEstimator 
import re
import pandas as pd 
import cv2
import matplotlib.pyplot as plt
import pickle
import bs4
import os

#for google  colabs
# from google.colab import drive
# drive.mount('/content/drive')
# # !ls "/content/drive/My Drive/Violin Code2"
# files = os.listdir()
# output = [dI for dI in os.listdir('/content/drive/My Drive/Violin Code2') if os.path.isdir(os.path.join("/content/drive/My Drive/Violin Code2",dI))]
os.listdir()
cwd = os.getcwd()
# print(cwd)
# ! ls

# In[5]:


#@title

#gets data into 
class Preprocessing:
#class of maker
    def __init__(self):
        #exception read in image
        self.tally_expetion_one=0
        #exception transfer greyscale image to jpg
        self.tally_expetion_two=0
        self.cities_df= pd.read_csv(r"C:\Users\alex2\Desktop\Violin\Violin Code\mathamaticacities.csv")
        self.cities_df.columns =['city', 'country', 'population']
        self.cities_df=self.cities_df.sort_values('population', ascending=False)
        self.cities_df = self.cities_df[~self.cities_df['population'].str.contains("\D")]       
        self.current_dir=currentDirectory=cwd = os.getcwd()
        self.directory_maker_list=os.listdir()
        self.maker_path_list=self.get_maker_paths()
        self.maker_class_list=self.get_maker_class_list()
        self.cities_csv='world-cities_csv.csv'
        self.cities_csv_df= pd.read_csv(os.path.join(self.current_dir,self.cities_csv))
        self.images=[]
        self.numpy_array=self.create_numpy_array()
        self.transfer_numerical_numpy_array_to_csv()
        self.save_image_text()

    def get_maker_paths(self):
        maker_list=[]
        for filename in self.directory_maker_list:   
            if "ViolinMaker" in filename:
                complete_path = os.path.join(self.current_dir, filename)
                maker_list.append(complete_path)
        return maker_list 

    def get_maker_class_list(self):
        maker_class_list=[]
        for filename in self.maker_path_list:   
            if "ViolinMaker" in filename:
                individual_maker_class=self.Maker(filename)
                maker_class_list.append(individual_maker_class)

        return maker_class_list
    def create_numpy_array(self, onlyViolinCello=False):
        instrument_outer_list=[]
        for maker_class in self.maker_class_list:         
            for instrument in maker_class.instrument_class_list:
                temp_dict=instrument.instrument_info_dict
                if 'USD' in temp_dict.keys():
                    instrument_info_list=self.filter_dict(temp_dict)
                    #only creates image list if instrument dict returns everything
                    if len(instrument_info_list) != 0:
                        photo_array=instrument.rgb_photos_array
                        if(len(photo_array)!=0):
                            if onlyViolinCello:
#                                 if instrument_info_list[1]=='Violin' or instrument_info_list[1]=='Cello':
                                if instrument_info_list[1]=='Violin':
                                    self.images.append(photo_array)
                                    instrument_outer_list.append(instrument_info_list)
                            else:
                                self.images.append(photo_array)
                                instrument_outer_list.append(instrument_info_list)
                            
        return numpy.array(instrument_outer_list)
        #gets all jpeg information
        
    #creates file all info

    def filter_dict(self, temp_dict):
        instrument_info_list=[]
        name=temp_dict["Maker Name"]
        instrument_type=temp_dict["Instrument Type"]
        location=self.convert_city_to_country(temp_dict["Location"])
        year_made=temp_dict["Year Made"]
        #gets potential values 99 percent missing
        old_sale_date=temp_dict['Old Date Sold']
        oldest_sale_date=temp_dict['Oldest Date Sold']
        old_USD=temp_dict['Old USD']
        oldest_USD = temp_dict['Oldest USD']
        city=temp_dict["Location"]
        if not re.match("[a-zA-Z]", city):
            city='error in city name'
        #checks if year_made is known or not, tries to convert year to just str(int)
        if type(year_made) is not int:
            if 'unknown' in year_made:
                year_made='-1'
            else:
                year_made_list=re.search('\d\d\d\d', year_made)
                if year_made_list:
                    year_made=year_made_list.group(0)
                else:
                    year_made='-1'
        #ensure sale date exists    
        date_sold=temp_dict["Sale Date"]
        if type(date_sold) is not int:
            if 'Series' in date_sold:
                return []
            date_sold=re.search('\d\d\d\d', date_sold).group(0)
        price_sold=temp_dict["USD"]
        #gets price or return -1
        price_numerical=self.convert_numerical_price(price_sold)
        #checks old sale date and oldest sale date
        if type(old_sale_date) is not int:
          try:
            if 'Series' not in old_sale_date:
              old_sale_date=re.search('\d\d\d\d', old_sale_date).group(0)
            else: old_sale_date="unknown"
          except: old_sale_date="unknown"
        if type(oldest_sale_date) is not int:
          try:
            if 'Series' not in oldest_sale_date:
              oldest_sale_date=re.search('\d\d\d\d', old_sale_date).group(0)
            else: oldest_sale_date="unknown"
          except: oldest_sale_date="unknown"
         #makes old price and oldest price int
        old_price_numerical=self.convert_numerical_price(old_USD)
        oldest_price_numerical=self.convert_numerical_price(oldest_USD)
        try:
            if price_numerical > 10 and int(date_sold) >1950:
                # price_with_inflation=self.calculate_inflated_price(price_numerical, date_sold)
                instrument_info_list.append(name)
                instrument_info_list.append(instrument_type)
                instrument_info_list.append(location)
                instrument_info_list.append(year_made)
                instrument_info_list.append(date_sold)
                instrument_info_list.append(price_numerical)
                #BEGIN add of mostly missing data for inflation calculations
                instrument_info_list.append(old_sale_date)
                instrument_info_list.append(old_price_numerical)
                instrument_info_list.append(oldest_sale_date)
                instrument_info_list.append(oldest_price_numerical)
                #Untested Code
                instrument_info_list.append(city)
                # instrument_info_list.append(price_with_inflation)
                return instrument_info_list
        except:
            print(" problem reading string")
        #return blank array if can't get price
        return []


    def convert_city_to_country(self, possible_city_name: str):
        country_names={'England':'United Kingdom'} 
#         print('finding country')
        if possible_city_name in country_names.keys():
            desired_country=country_names[possible_city_name]
#             print('found country {}' .format(desired_country) )
            return desired_country
        if possible_city_name in self.cities_df['country'].unique():
#             print('found country {}' .format(possible_city_name) )
            return possible_city_name
        try:
            series=self.cities_df.loc[self.cities_df['city'] == possible_city_name]
            most_populas=series.iloc[0, 1]
#             print('most populas matching city name is {}' .format(most_populas))
            return most_populas
        except:
            print('error')
            return "-1"
#         return possible_city_name

    def convert_numerical_price(self, price_sold_string):
        cost_parse_part_one=price_sold_string.split(" ")
        price_with_comma=cost_parse_part_one[-1]
        price_with_comma_no_dollar=price_with_comma.split('$')[-1]
        if re.match('\D', price_with_comma_no_dollar) is not None:
            return -1
        money_no_comma=int(price_with_comma_no_dollar.replace(',', ''))
        return money_no_comma

    def save_image_text(self):
        output = open('image_data.pkl', 'wb')
        pickle.dump(self.images, output)
        output.close()
    def transfer_numerical_numpy_array_to_csv(self):
            #TODO mayber change delimiter to ", "
            numpy.savetxt("dictionary_instruments_vetted.csv",  self.numpy_array, delimiter="," ,encoding='utf8',fmt='%s')

    class Maker(object):
        def __init__(self, path_maker):
            self.path_maker = path_maker
            #self.name=get_name()
            self.instrument_paths_list=self.get_instrument_paths_list()
            self.instrument_class_list=self.get_all_instruments()
        #not super important can skip for now
        def get_name(self):
            #TODO add interact with instrument return name
            print("Getting name of maker")

        def get_instrument_paths_list(self):
            instrument_path_list=[]
            path = os.getcwd()
            print(path)
            directory_list=os.listdir(self.path_maker)

            for filename in directory_list:   
                if "Instrument" in filename:
                    complete_path = os.path.join(self.path_maker, filename)
                    instrument_path_list.append(complete_path)
            return instrument_path_list

        def filler_method(self):
            print("END")

        #finds file with specific ty
        def find_file(self, pattern, path):
            result = []
            for root, dirs, files in os.walk(path):
                for name in files:
                    if fnmatch.fnmatch(name, pattern):
                        result.append(os.path.join(root, name))
            return result
        
        def get_all_instruments(self):
            instrument_class_list=[]
            for instrument_path in self.instrument_paths_list:
                individual_instrument_class=self.Instrument(instrument_path)
                instrument_class_list.append(individual_instrument_class)
            return instrument_class_list

        class Instrument(object):
            def __init__(self, path_instrument):
                self.path_instrument=path_instrument
                self.photos=self.get_photos()
                self.instrument_info_dict=self.get_instrument_info()
                self.rgb_photos_array=self.get_list_array_photos()
                

            def get_instrument_info(self):
                print("Getting dictionary about instrument")
                file_name='OrganizedViolinProfile.txt'
                organized_instrument_info_path = os.path.join(self.path_instrument, file_name)
                try:
                    f = open(organized_instrument_info_path, "r")
                    dict_text=f.read()
                    instrument_info= json.loads(dict_text) 
                    if(len(instrument_info)>0):
                        if len(self.photos)>0:
                            return instrument_info
                    #return if cant find any photos
                    return {}
                except OSError:
                    return {}

            def get_list_array_photos(self):
                
                inner_image=[]
                outer_image=numpy.zeros((SIZE_IMAGE, SIZE_IMAGE, 3), dtype="uint8")
                if(len(self.photos)>0):
                    for instrument_path in self.photos[:1]:
                        try:
                            img_array=cv2.imread(instrument_path, cv2.IMREAD_UNCHANGED)
                            new_array=cv2.resize(img_array, (SIZE_IMAGE, SIZE_IMAGE))
                            # plt.imshow(new_array)
                            # plt.show()
                            inner_image.append(new_array)
                        except Exception as exc:
                            print("inner image resize or read exception")
                            # self.tally_expetion_one+=1
                            pass
                        
                    try:
                        outer_image[0:SIZE_IMAGE, 0:SIZE_IMAGE] = inner_image[0]
#                         outer_image[0:64, 32:64] = inner_image[2]
#                         outer_image[0:300, 200:300] = inner_image[2]
                    # outer_image[256:512, 0:256] = inner_image[3]
                    except Exception as exc:
                        print("likely black and white image")
                        # super.super.self.tally_exception_two+=1
                        pass
                    return outer_image
                return[]
#     #                         outer_image[0:256, 0:256] = inner_image[0]
#                             print('returning image')
#                             return new_array
#                         except Exception as exc:
#                             print("exception occured in getting image")
#                             # self.tally_expetion_one+=1
#                             return []

#                 if(len(self.photos)>2):
#                     for instrument_path in self.photos[:3]:
#                         try:
#                             img_array=cv2.imread(instrument_path, cv2.IMREAD_UNCHANGED)
#                             #TODO change size of resize
#                             new_array=cv2.resize(img_array, (32, 64))
#                             # plt.imshow(new_array)
#                             # plt.show()
#                             inner_image.append(new_array)
#                         except Exception as exc:
#                             print("exception occured")
#                             # self.tally_expetion_one+=1
#                             pass
#                     try:                   
#                         #TODO resize to make more sense
#                         outer_image[0:64, 0:32] = inner_image[0]
#                         outer_image[0:64, 32:64] = inner_image[2]
# #                         outer_image[0:300, 200:300] = inner_image[2]
#                     # outer_image[256:512, 0:256] = inner_image[3]
#                     except Exception as exc:
#                         print("exception occured")
#                         print("likely black and white image")
#                         # super.super.self.tally_exception_two+=1
#                         pass
#                     return outer_image
#                 #if not 4 photos
#                 return []
            #only getting list of photo url paths at the second
            #TODO ask dad what is a better way, should store in a numpy array, or DF 
            def get_photos(self):
#                 print("Getting photos")
                directory_list=os.listdir(self.path_instrument)
                photo_path_list=[]
                for filename in directory_list:   
                     if filename.endswith(".jpg"):
                        file_path_photo=os.path.join(self.path_instrument, filename)
#                         print(file_path_photo)
                        photo_path_list.append(file_path_photo)
                return photo_path_list


# In[2]:


#runs code
instantiation_class=Preprocessing()
instrument_info=instantiation_class.numpy_array
#makes dataframe of code
df = pd.DataFrame(data=instrument_info)
images=instantiation_class.images

def convert_images(images):
    print(type(images))
    divided_images=[]
    for inner_image in images:
        divided_image=inner_image/255
        #TODO remove if cause issue
        divided_image=divided_image.tolist()
        divided_images.append(divided_image)
        # images = [images / 255 for x in images]
        # print(divided_images)
#     return numpy.array(divided_images)
    return divided_images
images=convert_images(images)
df["Image Array"]=images

df.head()


# In[7]:


df.iloc[:,0].unique().shape


# In[8]:


pd.set_option("display.max_rows", None, "display.max_columns", None)


# In[9]:


names=df.iloc[:, 0].unique()
instrument_type=df.iloc[:, 1].unique()
countries=df.iloc[:, 2].unique()
print(countries)


# In[10]:


found_price_df=df[df[7]!='-1']


# In[11]:


found_price_df[found_price_df.iloc[:,0]=='Antonio Stradivari']


# In[13]:


# found_price_df[3] = pd.to_numeric(found_price_df[3])
# found_price_df[4] = pd.to_numeric(found_price_df[4])
# found_price_df[5] = pd.to_numeric(found_price_df[5])
# found_price_df[6] = pd.to_numeric(found_price_df[6])
# found_price_df[7] = pd.to_numeric(found_price_df[7])
# found_price_df[8] = pd.to_numeric(found_price_df[8])


# In[14]:


def get_all_multisale_pairs(filtered_df):
    more_filtered_df=filtered_df[filtered_df[5]!=filtered_df[7]]
    return more_filtered_df


# In[15]:


more_filtered_df=get_all_multisale_pairs(found_price_df)


# In[18]:



def no_same(df):
    gains = df[(df[5] != df[7])
               &
               (df[5]*1.2 > df[7]) &
               (df[6]>1985)&
               (df[4]>1985)
               
             &  (df[5]>25)&
               ((df[4] >= df[6]+5) | (df[5] < (df[7] * 3)))&
               ((df[4] >= df[6]+3) | (df[5] < (df[7] * 2)))&
                (df[5] > (df[7] *.8) ) ]
    return gains
no_same=no_same(more_filtered_df)


# In[19]:


# # print(images.shape)
# # print(instantiation_class.images.shape)
# # ### Loading in Dow Stock data to compare inflation to Violins

# # In[17]:


# import numpy as np
# import requests
# text_stocks=requests.get('https://www.macrotrends.net/1319/dow-jones-100-year-historical-chart').text
# stocks_df=pd.read_html(text_stocks)[0]
# year_price_stock_df=stocks_df.iloc[:,0:2]['Dow Jones Industrial Average - Historical Annual Data']
# year_price_stock_df.iloc[:,1]=year_price_stock_df.iloc[:,1]*80
# plt.plot(year_price_stock_df.iloc[:,0].tolist(),year_price_stock_df.iloc[:,1].tolist())


# # In[18]:

# # plt.show()
# def convert_and_plot_pairs(dataframe):
# #     x1_cello=dataframe[dataframe[1]=='Cello'][6].to_list()
# #     y1_cello=dataframe[dataframe[1]=='Cello'][7].to_list()
# #     x2_cello=dataframe[dataframe[1]=='Cello'][4].to_list()
# #     y2_cello=dataframe[dataframe[1]=='Cello'][5].to_list()
    
# #     x1_violin=dataframe[dataframe[1]=='Violin'][6].to_list()
# #     y1_violin=dataframe[dataframe[1]=='Violin'][7].to_list()
# #     x2_violin=dataframe[dataframe[1]=='Violin'][4].to_list()
# #     y2_violin=dataframe[dataframe[1]=='Violin'][5].to_list()
    
#     x1=dataframe[6].to_list()
#     y1=dataframe[7].to_list()
#     x2=dataframe[4].to_list()
#     y2=dataframe[5].to_list()
    
# #     for i in range(0, len(x1_cello)): 
# #         x1_cello[i] = int(x1_cello[i]) 
# #     for i in range(0, len(x2_cello)): 
# #         x2_cello[i] = int(x2_cello[i]) 
# #     for i in range(0, len(y1_cello)): 
# #         y1_cello[i] = int(y1_cello[i]) 
# #     for i in range(0, len(y2_cello)): 
# #         y2_cello[i] = int(y2_cello[i]) 
        
# #     for i in range(0, len(x1_violin)): 
# #         x1_violin[i] = int(x1_violin[i]) 
# #     for i in range(0, len(x2_violin)): 
# #         x2_violin[i] = int(x2_violin[i]) 
# #     for i in range(0, len(x1_violin)): 
# #         x1_violin[i] = int(x1_violin[i]) 
# #     for i in range(0, len(y2_violin)): 
# #         y2_violin[i] = int(y2_violin[i])       
        
#     for i in range(0, len(x1)): 
#         x1[i] = int(x1[i]) 
#     for i in range(0, len(x2)): 
#         x2[i] = int(x2[i]) 
#     for i in range(0, len(y1)): 
#         y1[i] = int(y1[i]) 
#     for i in range(0, len(y2)): 
#         y2[i] = int(y2[i]) 
        
# #     lists_cello=[x1_cello,y1_cello,x2_cello,y2_cello]
# #     point1_cello = [x1_cello, y1_cello]
# #     point2_cello = [x2_cello, y2_cello]
# #     x_values_cello = [point1_cello[0], point2_cello[0]]
# #     y_values_cello = [point1_cello[1], point2_cello[1]]
    
# #     lists_violin=[x1_violin,y1_violin,x2_violin,y2_violin]
# #     point1_violin = [x1_violin, y1_violin]
# #     point2_violin = [x2_violin, y2_violin]
# #     x_values_violin = [point1_violin[0], point2_violin[0]]
# #     y_values_violin = [point1_violin[1], point2_violin[1]]
    
    
# #     print(x1)
# #     print(x2)
#     lists=[x1,y1,x2,y2]
#     point1 = [x1, y1]
#     point2 = [x2, y2]
#     x_values = [point1[0], point2[0]]
#     y_values = [point1[1], point2[1]]
#     plt.xlabel('Year Sold') 
#     plt.ylabel('Price at Auction Pairs')
#     plt.title('Price Changes in Violins\nSold Multiple Times')
#     plt.plot(x_values, y_values)
# #     plt.plot(x_values_cello, y_values_cello, color='orange', label='Cello')
# #     plt.plot(x_values_violin, y_values_violin, color='blue', label='Violin')
#     plt.ticklabel_format(style='plain', axis='y')
# #     year_price_stock_df.plot(x="Year", y=["AverageClosing Price"])
#     plt.plot(year_price_stock_df.iloc[0:50,0].tolist(),year_price_stock_df.iloc[0:50,1].tolist(), 'r')
#     plt.savefig('Price Changes in Violins Sold Multiple Times.png')
#     plt.legend()
#     plt.show()
#     return lists


# # In[19]:


# all_lists=convert_and_plot_pairs(no_same)


# # In[20]:


# In[22]:


import numpy as np
import requests
import random

def convert_and_plot_pairs(dataframe):     
    text_stocks=requests.get('https://www.macrotrends.net/1319/dow-jones-100-year-historical-chart').text
    stocks_df=pd.read_html(text_stocks)[0]
    year_price_stock_df=stocks_df.iloc[:,0:2]['Dow Jones Industrial Average - Historical Annual Data']
    year_price_stock_df.iloc[:,1]=year_price_stock_df.iloc[:,1]*dataframe[5].max()/50000
    
    x1=dataframe[6].to_list()
    y1=dataframe[7].to_list()
    x2=dataframe[4].to_list()
    y2=dataframe[5].to_list()
    
    for i in range(0, len(x1)): 
        x1[i] = int(x1[i]) 
    for i in range(0, len(x2)): 
        x2[i] = int(x2[i]) 
    for i in range(0, len(y1)): 
        y1[i] = int(y1[i]) 
    for i in range(0, len(y2)): 
        y2[i] = int(y2[i]) 
        
    lists=[x1,y1,x2,y2]
    point1 = [x1, y1]
    point2 = [x2, y2]
    x_values = [point1[0], point2[0]]
    y_values = [point1[1], point2[1]]
    plt.xlabel('Year Sold') 
    plt.ylabel('Price at Auction Pairs')
    keys=dataframe[1].unique()
    instrument_type='Instruments'
    if len(keys)==1:
        instrument_type=key[1]
    plt.title('Price Changes for All Instruments \nSold Multiple Times'.format(instrument_type))
    ax=plt.plot(x_values, y_values)
    plt.ticklabel_format(style='plain', axis='y')
#     year_price_stock_df.plot(x="Year", y=["AverageClosing Price"])
    plt.plot(year_price_stock_df.iloc[0:50,0].tolist(),year_price_stock_df.iloc[0:50,1].tolist(), 'black', label='Nasdaq with scaled constant')
    plt.legend()
    num=random.randint(1, 200)
    plt.savefig('Price Changes in Violins Sold Multiple Times{}.png'.format(str(num)))
    plt.show()
    return lists

all_lists=convert_and_plot_pairs(no_same)


# In[23]:


# no_same[1].unique()


# In[ ]:


import math
from statistics import median
# ## Inflation Calculations

def find_average_inflation(list_dates_and_price):
    r_multiplied_by_num_years=0
    num_years=0
    for i in range(len(list_dates_and_price[0])):
        A=1.0*list_dates_and_price[3][i]
        P=1.0*list_dates_and_price[1][i]
        if P == 0:
#             print('coult not find auction price at index {0}'.format(i))
            continue
        numerator=math.log(A/P)
#         print(numerator)
        r_multiplied_by_num_years+=numerator
        year_diff=list_dates_and_price[2][i]-list_dates_and_price[0][i]
#         print(numerator/year_diff)
        num_years+=year_diff    
    if num_years==0:
        print('no information found for query')
        return -1
    return r_multiplied_by_num_years/num_years        

def find_med_inflation(list_dates_and_price):
    inflation_nums=[]
    r_multiplied_by_num_years=0
    num_years=0
    for i in range(len(list_dates_and_price[0])):
        A=1.0*list_dates_and_price[3][i]
        P=1.0*list_dates_and_price[1][i]
        if P == 0:
#             print('coult not find auction price at index {0}'.format(i))
            continue
        numerator=math.log(A/P)
#         print(numerator)
        r_multiplied_by_num_years+=numerator
        year_diff=list_dates_and_price[2][i]-list_dates_and_price[0][i]
#         print(numerator/year_diff)
        num_years+=year_diff
        inflation_nums.append(r_multiplied_by_num_years/num_years)
    if num_years==0:
        print('no information found for query')
        return -1
    return median(inflation_nums)

def inflation_nums(list_dates_and_price):
    inflation_nums=[]
    r_multiplied_by_num_years=0
    num_years=0
    for i in range(len(list_dates_and_price[0])):
        A=1.0*list_dates_and_price[3][i]
        P=1.0*list_dates_and_price[1][i]
        if P == 0:
#             print('coult not find auction price at index {0}'.format(i))
            continue
        numerator=math.log(A/P)
#         print(numerator)
        r_multiplied_by_num_years+=numerator
        year_diff=list_dates_and_price[2][i]-list_dates_and_price[0][i]
#         print(numerator/year_diff)
        num_years+=year_diff
        inflation_nums.append(r_multiplied_by_num_years/num_years)
    if num_years==0:
        print('no information found for query')
        return -1
    return inflation_nums

df[3] = pd.to_numeric(df[3])
df[4] = pd.to_numeric(df[4])
df[5] = pd.to_numeric(df[5])
# df[6] = pd.to_numeric(df[6])


dataframe_filtered_dictionary={}
dataframe_filtered_dictionary['Cello']=no_same[no_same.iloc[:,1]=='Cello']
dataframe_filtered_dictionary['Violin']=no_same[no_same.iloc[:,1]=='Violin']
dataframe_filtered_dictionary['Viola']=no_same[no_same.iloc[:,1]=='Viola']
dataframe_filtered_dictionary['Italy']=no_same[no_same.iloc[:,2]=='Italy']
dataframe_filtered_dictionary['France']=no_same[no_same.iloc[:,2]=='France']
dataframe_filtered_dictionary['Contemporary']=no_same[no_same.iloc[:,3]>1920]
dataframe_filtered_dictionary['Old']=no_same[no_same.iloc[:,3]<1800]
dataframe_filtered_dictionary['Trash or Trade']=no_same[no_same.iloc[:,5]<10000]
dataframe_filtered_dictionary['Proffesional']=no_same[(no_same.iloc[:,5]>10000)&(no_same.iloc[:,5]<50000)]
dataframe_filtered_dictionary['Collectors']=no_same[no_same.iloc[:,5]>50000]
# dataframe_filtered_dictionary['World Class']=no_same[no_same.iloc[:,5]>200000]
# dataframe_filtered_dictionary['Italian Cellos']=no_same[(no_same.iloc[:,2]=='Italy') & (no_same.iloc[:,1]=='Cello' )]
# dataframe_filtered_dictionary['Old Italian Cellos']=no_same[(no_same.iloc[:,2]=='Italy') & (no_same.iloc[:,1]=='Cello' )& (no_same.iloc[:,3]<1800 )]
# dataframe_filtered_dictionary['1860 to 1950 Italian Cellos']=no_same[(no_same.iloc[:,2]=='Italy') & (no_same.iloc[:,1]=='Cello' )& (no_same.iloc[:,3]>1880 ) &(no_same.iloc[:,3]<1950 )]


# In[24]:


# dataframe_filtered_dictionary['Contemporary Italian Cellos']
general_inflation=find_average_inflation(all_lists)
print('Mean Inflation for all instruments is {0}'.format(general_inflation))
gen_med_inflation=find_med_inflation(all_lists)
gen_inflation_list=inflation_nums(all_lists)


# In[27]:


gen_med_inflation


# In[29]:


print('Median Inflation for is {}'.format(gen_med_inflation))
plt.hist(gen_inflation_list, alpha=0.5, bins=20)
plt.show()
print()
print()
import scipy.stats as st
plt.hist(gen_inflation_list, density=True, bins=50, label="Inflation Distribution All Instruments")
plt.xlim(0, .2)
kde_xs = np.linspace(0, .2, 301)
kde = st.gaussian_kde(gen_inflation_list)
# plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
# plt.legend(loc="upper left")
# plt.ylabel('Probability')
# plt.xlabel('Inflation Calculation for Each Instrument')
# plt.title("Histogram")
# plt.show()


# In[30]:


num=1
for key in dataframe_filtered_dictionary:
    print(key)
    lists=convert_and_plot_pairs(dataframe_filtered_dictionary[key])
    inflation=find_average_inflation(lists)
    med_inflation=find_med_inflation(lists)
    inflation_list=inflation_nums(lists)
    print('Mean Inflation for {0} is {1}'.format(key, inflation))
    print('Median Inflation for {0} is {1}'.format(key, med_inflation))
    plt.hist(inflation_list, density=True, bins=50, label="Inflation Distribution Specific Instruments")
    plt.xlim(0, .2)
    kde_xs = np.linspace(0, .2, 301)
    kde = st.gaussian_kde(inflation_list)
    plt.plot(kde_xs, kde.pdf(kde_xs), label="PDF")
    plt.legend(loc="upper left")
    plt.ylabel('Probability')
    plt.xlabel('Inflation Calculation for Each Instrument')
    plt.title("Histogram")
    plt.show()
    print()
    print()


# In[31]:


inflation=find_average_inflation(all_lists)


# In[32]:


print(inflation)


# In[33]:


#gets auction data
multip_times=df[5]*(1+inflation)**(2020-df[4])


# In[34]:


list_of_ints = []
for item in multip_times:
    if int(item)<0:
        item=-1
    list_of_ints.append(int(item))
print(len(list_of_ints))
print(df.shape)


# In[36]:


df['Price with Inflation']=list_of_ints


# In[37]:


df.head()


# In[38]:


found_price_df.iloc[:,5].sum()


# In[39]:


df['Price with Inflation'].sum()


# In[40]:


df[3].astype('int')
df[4].astype('int')
df[5].astype('int')
df[7].astype('int')
df[9].astype('int')
df.dtypes


# In[44]:


# df.loc[df[3]==-1,df[3]]=2021
df[3].replace({-1: 2021})
df[3].median()


# In[49]:


df.plot(x =4, y=5, kind = 'scatter', title ='Price Sold of Instruments over Time')


# In[50]:


df.plot(x =4, y='Price with Inflation', kind = 'scatter',title ='Value of Instruments over Time')


# In[51]:


names=df.iloc[:, 0].unique()
instrument_type=df.iloc[:, 1].unique()
countries=df.iloc[:, 2].unique()


# In[52]:


print(countries)


# In[53]:


df.head()
#check df head: if include city, and no glaring error modify lines below,
#else modify lines above


# In[54]:


df.shape


# In[55]:


sorted_df_population=df.sort_values(by=[3])


# In[56]:


sorted_df_population.head()


# In[57]:


sorted_df_population.to_csv('dataframe_with_cities_images_everything.csv', index=False)


# In[58]:


def get_city_years_clumps(dataframe_needed):
    print('to_implement')
    city=dataframe_needed[10].to_list()
    year_of_city=dataframe_needed[3].to_list()
    return [city, year_of_city]


# In[59]:


city_year_data=get_city_years_clumps(sorted_df_population)


# In[62]:


outer_list_city_clumps=[]
CLUMP_SIZE=50
start_year=1550
inner_list=[]
for itr in range(len(city_year_data[0])):
    city=city_year_data[0][itr]
    year=city_year_data[1][itr]
    if year>=start_year:
        print('adding city {}'.format(city))
        inner_list.append(city)
        if year> start_year+CLUMP_SIZE:
            print('append list to outerlist and create new empty list')
            outer_list_city_clumps.append(inner_list)
            start_year=start_year+CLUMP_SIZE
            inner_list=[]            

        #create code that creates clumps of every 20 years


# In[63]:


outer_list_city_clumps
df_city_clumps = pd.DataFrame(outer_list_city_clumps)
df_city_clumps.head()


# In[64]:


# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np
import glob
import cv2
import os
from sklearn.model_selection import train_test_split

df_model_ready=df.copy()
df_model_ready=df_model_ready.drop(columns=[4, 5,6,7,8,9,10])
df_model_ready.columns = ["Maker Name", "Instrument Type", "Country Made", "Year Made", "Current Value", "Image Array"]
# df_model_ready.columns = ["Maker Name", "Instrument Type", "Country Made", "Year Made", "Current Value"]
print(df_model_ready.iloc[0:-1].shape)

# in instruments with missing value, search for similar info of same maker
def get_unique_fillers_for_missing_value(inner_info):
    name=inner_info[0]
    print(name)
    instr=inner_info[1]
    country=inner_info[2]
    year_made=inner_info[3]
    value=inner_info[5]
    photo_arr=inner_info[4]
    match_df=df_model_ready.loc[df_model_ready['Maker Name'] == name]
    if country == '-1' or country == -1:
        try:
#             print('in try loop find country')
#             print('found match df')
            poss_countries=match_df['Country Made'].unique()
#             print(poss_countries)
            location_countries = np.where(poss_countries!= '-1')
            country=poss_countries[location_countries].flat[0]
        except: 
            print('cant find country')
    if year_made == -1 or year_made =='-1':
        try:
            year_made=match_df['Year Made'].median()
        except: 
            print('cant find country')
            year_made=-1

    # print('values is {}'.format(value))
    if value == -1 or value =='-1':    
        try:
            value=match_df['Current Value'].median()
#             print(5)
        except: 
            print('cant find value')
#     print('Result')
#     print([name, instr, country, year_made, value, photo_arr])             
    return [name, instr, country, year_made, value, photo_arr]   

df_to_list=df_model_ready.values
new_list=[]
for inner in df_to_list:
    x=inner.tolist()
#     print('inner list is')
#     print(x)
#     print(x[4].shape)
    new_list.append(get_unique_fillers_for_missing_value(x))    

df3=pd.DataFrame(new_list) 
df3.columns = ["Maker Name", "Instrument Type", "Country Made", "Year Made", "Current Value", "Image Array"]


# In[65]:


# ### Makes all instruments Italian, French or Unkown

def get_rid_uncommon_keys(df):
    outer_list = df.values.tolist()
    key_freq = {}
    #count frequencies
    for inner in outer_list:
        if inner[2] not in key_freq:
            key_freq[inner[2]] = 0 
        else:
            key_freq[inner[2]] += 1
    #sort keys
    from collections import Counter 
    k = Counter(key_freq) 
    high = k.most_common(DESIRED_COUNTRIES)
    def Extract(lst): 
        return [item[0] for item in lst] 
    desired_keys=Extract(high)
    #now loop through and replace all keys to -1 if cannot find key
    df_list=[]
    for inner in outer_list:
        if inner[2] in desired_keys:
            df_list.append(inner)
        #TODO check if make string or int
        else: 
            inner[2] = '-1'
            df_list.append(inner)
    # return pd.DataFrame(data=df_list, columns = ["Maker Name", "Instrument Type", "Country Made", "Year Made", "Current Value"])
    return pd.DataFrame(data=df_list, columns = ["Maker Name", "Instrument Type", "Country Made", "Year Made", "Current Value", "Image Array"])
df3=get_rid_uncommon_keys(df3)
np.where(df3['Year Made'] == '-1', '2021', '1999') 
df3['Year Made'] = np.where(df3['Year Made']==-1, 2021, df3['Year Made'])
# np.where(array1==0, 1, array1) 
df3.head()


# In[ ]:


df4=df3.copy()
# df4["Maker Name"] = df4["Maker Name"].astype('category')
# df4["Instrument Type"] = df4["Instrument Type"].astype('category')
# df4["Year Made"] = df4["Year Made"].astype('int64')
# df4["Country Made"] = df4["Country Made"].astype('category')
df4.dtypes
X=df4.drop('Maker Name', axis='columns')
X.head()
import pickle
with open('image_data.pkl', 'rb') as f:
    images = pickle.load(f)
import copy
preprocessed_images=copy.deepcopy(images)


# In[66]:


def view_instrument_and_info(number):
    img = Image.fromarray(preprocessed_images[number], 'RGB')
    img.show()
    print(df3.iloc[number,:])


# In[68]:


import pickle 
X.to_pickle("/Users/alex2/Desktop/Violin/Violin Code GPU Run/X_dataframe_post_processing.pkl")

