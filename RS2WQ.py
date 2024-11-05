# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 12:51:06 2024

@author: mShen
"""

import numpy as np
import os
import glob
from osgeo import gdal
import joblib
from skimage.morphology import erosion
import warnings
warnings.filterwarnings("ignore")


#functions
def read_img(filename):
    dataset=gdal.Open(filename)       #打开文件

    im_width = dataset.RasterXSize    #栅格矩阵的列数
    im_height = dataset.RasterYSize   #栅格矩阵的行数

    im_geotrans = dataset.GetGeoTransform()  #仿射矩阵
    im_proj = dataset.GetProjection() #地图投影信息
    im_data = dataset.ReadAsArray(0,0,im_width,im_height) #将数据写成数组，对应栅格矩阵
    
    
    if im_geotrans[5]>0:
        im_geotrans = list(im_geotrans)
        im_geotrans[5] = -im_geotrans[5]
        im_geotrans[3]=im_geotrans[3] - im_geotrans[5] * im_height
        im_geotrans = tuple(im_geotrans)
        for i in range(im_data.shape[0]):
            im_data[i]=np.flipud(im_data[i])
    
    del dataset
    return im_proj,im_geotrans,im_data,im_height,im_width
def write_img(filename,im_data,im_proj=None,im_geotrans=None):
        #gdal数据类型包括
        #gdal.GDT_Byte, 
        #gdal .GDT_UInt16, gdal.GDT_Int16, gdal.GDT_UInt32, gdal.GDT_Int32,
        #gdal.GDT_Float32, gdal.GDT_Float64

        #判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32

        #判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1,im_data.shape 

        #创建文件
        driver = gdal.GetDriverByName("GTiff")            #数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)

        if im_geotrans is not None:
            dataset.SetGeoTransform(im_geotrans)              #写入仿射变换参数
        if im_proj is not None:
            dataset.SetProjection(im_proj)                    #写入投影

        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  #写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i+1).WriteArray(im_data[i])
        driver=None
        dataset=None
        del dataset,driver
        
def make_input(im_data,n_features,mask):
    #glint corr.
    R865=im_data[8,:,:]
    R865[R865<0]=0
    R1610=im_data[9,:,:]
    R1610[R1610<0]=0
    R_glint=(R865<R1610)*R865+(R865>=R1610)*(R1610)
    #2202
    R2202=im_data[10,:,:]
    R865[R865<0]=0
    R_glint=(R_glint<R2202)*R_glint+(R_glint>=R2202)*(R2202)
    R_glint=R_glint/np.pi
    R_glint[R_glint<0]=0
    # R_glint=0
    Rrs=(im_data[0:11,:,:]/np.pi-R_glint)*mask
    Rrs[np.isnan(Rrs)]=0
    
    X=np.zeros((n_features,im_data.shape[1],im_data.shape[2]))
    X[0:6,:,:]=Rrs[0:6,:,:]
    # X[6,:,:]=im_data[8,:,:]
    # X[7,:,:]=im_data[9,:,:]
    num=6
    idx1,idx2=0,2#443/560
    imd=Rrs[idx1,:,:]/Rrs[idx2,:,:]
    imd[Rrs[idx1,:,:]<1e-5]=0
    imd[Rrs[idx2,:,:]<1e-5]=0
    X[num,:,:]=imd
    
    num=num+1
    idx1,idx2=1,2#492/560
    imd=Rrs[idx1,:,:]/Rrs[idx2,:,:]
    imd[Rrs[idx1,:,:]<1e-5]=0
    imd[Rrs[idx2,:,:]<1e-5]=0
    X[num,:,:]=imd
    
    num=num+1
    idx1,idx2=3,2#665/560
    imd=Rrs[idx1,:,:]/Rrs[idx2,:,:]
    imd[Rrs[idx1,:,:]<1e-5]=0
    imd[Rrs[idx2,:,:]<1e-5]=0
    X[num,:,:]=imd
    
    num=num+1
    idx1,idx2=4,2#704/560
    imd=Rrs[idx1,:,:]/Rrs[idx2,:,:]
    imd[Rrs[idx1,:,:]<1e-5]=0
    imd[Rrs[idx2,:,:]<1e-5]=0
    X[num,:,:]=imd
    
    num=num+1
    idx1,idx2=4,3#704/665
    imd=Rrs[idx1,:,:]/Rrs[idx2,:,:]
    imd[Rrs[idx1,:,:]<0.005]=0
    imd[Rrs[idx2,:,:]<0.005]=0
    X[num,:,:]=imd
    
    
    return X,Rrs
def Rrs2OACs():
    tag='Chla'
    outname=os.path.join(outdir,os.path.basename(imgname).replace('BOA',tag))
    models=[joblib.load(rs) for rs in glob.glob(os.path.join(modeldir,f'*{tag}-SEN2COR.pkl'))]
    X,Rrs=make_input(im_data,len(models[0].feature_importances_),mask)
    X,Rrs=X*mask,Rrs*mask
    band,line,sample =X.shape[0], X.shape[1],X.shape[2]
    x = X.reshape(band,-1).T
    x[np.isnan(x)]=0
    x[np.isinf(x)]=0
    mask_idx=(mask.flatten()==True)
    xt=x[mask_idx,:]
    fres=[]
    for rfr in models:
        fres+=[rfr.predict(xt)]
    fres=(np.array(fres))
    fres[fres==0]=np.nan    
    fres=np.nanmean(fres,axis=0)
    imgRES=np.zeros((line,sample)).flatten()
    imgRES[mask_idx]=fres
    imgRES=imgRES.reshape(line,sample)*mask
    imgRES[np.isnan(imgRES)]=0
    chla_data=imgRES
    
    tag='SPM'
    outname=os.path.join(outdir,os.path.basename(imgname).replace('BOA',tag))
    models=[joblib.load(rs) for rs in glob.glob(os.path.join(modeldir,f'*{tag}-SEN2COR.pkl'))]
    X,Rrs=make_input(im_data,len(models[0].feature_importances_),mask)
    X,Rrs=X*mask,Rrs*mask
    band,line,sample =X.shape[0], X.shape[1],X.shape[2]
    x = X.reshape(band,-1).T
    x[np.isnan(x)]=0
    x[np.isinf(x)]=0
    mask_idx=(mask.flatten()==True)
    xt=x[mask_idx,:]
    fres=[]
    for rfr in models:
        fres+=[rfr.predict(xt)]
    fres=(np.array(fres))
    fres[fres==0]=np.nan    
    fres=np.nanmean(fres,axis=0)
    imgRES=np.zeros((line,sample)).flatten()
    imgRES[mask_idx]=fres
    imgRES=imgRES.reshape(line,sample)*mask
    imgRES[np.isnan(imgRES)]=0
    spm_data=imgRES
    
    tag='ag440'
    outname=os.path.join(outdir,os.path.basename(imgname).replace('BOA',tag))
    models=[joblib.load(rs) for rs in glob.glob(os.path.join(modeldir,f'*{tag}-SEN2COR.pkl'))]
    X,Rrs=make_input(im_data,len(models[0].feature_importances_),mask)
    X,Rrs=X*mask,Rrs*mask
    band,line,sample =X.shape[0], X.shape[1],X.shape[2]
    x = X.reshape(band,-1).T
    x[np.isnan(x)]=0
    x[np.isinf(x)]=0
    mask_idx=(mask.flatten()==True)
    xt=x[mask_idx,:]
    fres=[]
    for rfr in models:
        fres+=[rfr.predict(xt)]
    fres=(np.array(fres))
    fres[fres==0]=np.nan    
    fres=np.nanmean(fres,axis=0)
    imgRES=np.zeros((line,sample)).flatten()
    imgRES[mask_idx]=fres
    imgRES=imgRES.reshape(line,sample)*mask
    imgRES[np.isnan(imgRES)]=0
    ag_data=imgRES
    
    tag='SDD'
    outname=os.path.join(outdir,os.path.basename(imgname).replace('BOA',tag))
    models=[joblib.load(rs) for rs in glob.glob(os.path.join(modeldir,f'*{tag}-SEN2COR.pkl'))]
    X,Rrs=make_input(im_data,len(models[0].feature_importances_),mask)
    X,Rrs=X*mask,Rrs*mask
    band,line,sample =X.shape[0], X.shape[1],X.shape[2]
    x = X.reshape(band,-1).T
    x[np.isnan(x)]=0
    x[np.isinf(x)]=0
    mask_idx=(mask.flatten()==True)
    xt=x[mask_idx,:]
    fres=[]
    for rfr in models:
        fres+=[rfr.predict(xt)]
    fres=(np.array(fres))
    fres[fres==0]=np.nan    
    fres=np.nanmean(fres,axis=0)
    imgRES=np.zeros((line,sample)).flatten()
    imgRES[mask_idx]=fres
    imgRES=imgRES.reshape(line,sample)*mask
    imgRES[np.isnan(imgRES)]=0
    sdd_data=imgRES
    
    return chla_data,spm_data,ag_data,sdd_data
        
# main        
if __name__ == "__main__":
    
    imgname=r'./testdata/LakeID_F32A001_20220227_BOA.tif'
    modeldir=r'./Model'
    outdir=r'./output'
    if not os.path.exists(outdir): os.makedirs(outdir)
    #read data
    im_proj,im_geotrans,im_data,im_height,im_width=read_img(imgname)
    im_data=im_data*1.0
    im_data[im_data==-2147483648]=np.nan
    im_data[0:11,:,:]=(im_data[0:11,:,:]-1)*0.0001

    #mask
    water=im_data[14,:,:]==6
    #cloud
    noncloud1=im_data[13,:,:]==0
    noncloud2=im_data[12,:,:]<=20
    #bloom
    FAI=im_data[8,:,:]-(im_data[3,:,:]+(im_data[9,:,:]-im_data[3,:,:])*(864-665)/(1610-665))
    nonbloom=FAI<0.005
    
    mask=water*noncloud1*noncloud2*nonbloom
    
    #mask向内腐蚀2像元
    cross1 = np.array([
                      [0,0,1,0,0],#1
                      [0,1,1,1,0],#2
                      [1,1,1,1,1],#3
                      [0,1,1,1,0],#4
                      [0,0,1,0,0],#5
                    ])
    eroded_mask = erosion(mask, cross1)
    eroded_mask[eroded_mask<1]=0
    mask=eroded_mask    
    
    #Rrs->OACs
    chla_data,spm_data,ag_data,sdd_data=Rrs2OACs()
    
    #OACs->WQC
    tag='WQC'
    outname=os.path.join(outdir,os.path.basename(imgname).replace('BOA',tag))
    models=[joblib.load(rs) for rs in glob.glob(os.path.join(modeldir,f'*{tag}-SEN2COR.pkl'))]
    mask=(chla_data>0)*(spm_data>0)*(ag_data>0)*(sdd_data>0)
    X=np.array([chla_data,spm_data,ag_data,sdd_data])
    x = X.reshape(4,-1).T
    x[np.isnan(x)]=0
    x[np.isinf(x)]=0
    
    mask_idx=(mask.flatten()==True)
    xt=x[mask_idx,:]         
    fres=[]
    for rfr in models:
        fres+=[rfr.predict(xt)+1]
    fres=(np.array(fres)) 
    fres=np.nanmean(fres,axis=0)
    fres[fres>=1.5]=2.0
    fres[fres<1.5]=1.0
    fres[fres==0]=np.nan    
    imgRES=np.zeros((im_height,im_width)).flatten()
    imgRES[mask_idx]=fres
    imgRES=imgRES.reshape(im_height,im_width)*mask
    imgRES[np.isnan(imgRES)]=0
    write_img(outname,imgRES,im_proj=im_proj,im_geotrans=im_geotrans)
    
    
    
    
    