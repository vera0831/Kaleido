#include "RcppArmadillo.h"
using namespace arma;


//image value bound limit
// [[Rcpp::export]]
arma::cube clamping(arma::cube img){
  for(int i = 0;i<img.n_slices;i++){
    for(int j = 0; j < img.n_cols;j++){
      for(int k = 0; k < img.n_rows; k++){
        if(img(k,j,i)>1){
          img(k,j,i) = 1;
        }else if(img(k,j,i)<0){
          img(k,j,i) = 0;
        }
      }
    }
  }
  
  return img;
}


//height and width of image

// [[Rcpp::export]]
int im_height(arma::cube img){
  return img.n_rows;
}


// [[Rcpp::export]]
int im_width(arma::cube img){
  return img.n_cols;
}


//how many layers of the image

// [[Rcpp::export]]
int im_nc(arma::cube img){
  return img.n_slices;
}


//how  many points of the image
// [[Rcpp::export]]
int im_npix(arma::cube img){
  return img.n_elem;
}

//if the image only have one layer,
//then repeat that layer to a three channls image
// [[Rcpp::export]]
arma::cube im_rep(arma::mat img){
    arma::cube outimg(img.n_rows,img.n_cols,3);
  outimg.slice(0) = img;
  outimg.slice(1) = img;
  outimg.slice(2) = img;
  
  return outimg;
}


//image style 1 , feather
// [[Rcpp::export]]
arma::cube feather(arma::cube img){
    arma::cube A = 1-img;
  int slices = A.n_slices;
  int row = A.n_rows;
  int col = A.n_cols;
  double center = pow(row/2,2) + pow(col/2,2);
    arma::cube outA(row,col,slices,fill::zeros);
  int i,j,k;
  for(i = 0;i<slices;i++){
    for(j = 0;j<col;j++){
      for(k = 0;k<row;k++){
        outA(k,j,i) = A(k,j,i)*(1-(pow(j-col/2,2)+pow(i-row/2,2))/center);
      }
    }
  }
  return 1-outA;
}


//image style 2 : nostalgia
// [[Rcpp::export]]
arma::cube nostalgia(arma::cube img){
  int slices = img.n_slices;
  int row = img.n_rows;
  int col = img.n_cols;
    arma::cube outimg(row,col,slices,fill::zeros);
  int i,j;
  for(j = 0;j<col;j++){
    for(i = 0;i<row;i++){
      outimg(i,j,0) = img(i,j,0)*0.393+img(i,j,1)*0.769+0.189*img(i,j,2);
      outimg(i,j,1) = img(i,j,0)*0.349+img(i,j,1)*0.686+0.168*img(i,j,2);
      outimg(i,j,2) = img(i,j,0)*0.272+img(i,j,1)*0.543+0.131*img(i,j,2);
    }
  }
  return outimg;
}


//image style 3: lighting
// [[Rcpp::export]]
arma::cube lighting(arma::cube img,double centerx = 0.5, double centery = 0.5,double strength = 0.5){
  int slices = img.n_slices;
  int row = img.n_rows;
  int col = img.n_cols;
  
    arma::cube outimg(row,col,slices,fill::zeros);
  
  int centerX = col * centerx;
  int centerY = row * centery;
  double radius;
  //double radius = min(centerX, centerY);
  if(centerX < centerY){
     radius = centerX;
  }else{
     radius = centerY;
  }
  
  for(int i = 0; i < row; i++){
    for(int j = 0; j < col; j++){
      double distance = pow((i-centerY),2) + pow((j-centerX), 2);
      //‘≠ ºRGB÷µ
      double R = img(i,j,0);
      double G = img(i,j,1);
      double B = img(i,j,2);
      
      if(distance < radius*radius){
        double result = strength * (1.0 - sqrt(distance)/radius);
        R = R + result;
        G = G + result;
        B = B + result;
      }
      outimg(i,j,0) = R;
      outimg(i,j,1) = G;
      outimg(i,j,2) = B;
    }
  }
  
  return outimg;
}


//image style 4 : fleetingtime
// [[Rcpp::export]]
arma::cube fleetingtime(arma::cube img){
  int slices = img.n_slices;
  int row = img.n_rows;
  int col = img.n_cols;
  

    arma::cube outimg(row,col,slices,fill::zeros);
  double R,G,B;
  for(int i = 0; i < row; i++){
    for(int j = 0; j < col; j++){

      R = sqrt(img(i,j,0))*0.75;
      G = img(i,j,1);
      B = img(i,j,2);
      outimg(i,j,0) = R;
      outimg(i,j,1) = G;
      outimg(i,j,2) = B;
    }
  }
  
  return outimg;
}



arma::mat gaussfiltterkernel(int ksize,double sigma){
    arma::mat outkenal(ksize,ksize);
  
  double sum = 0;
  
  int center_i = ksize/2,center_j = ksize/2;
  for(int i = 0; i < ksize; i++){
    for(int j = 0; j < ksize; j++){
      outkenal(i,j) =exp( -1.0*(pow(i-center_i,2)+pow(j-center_j,2))/(2.0 * pow(sigma, 2)));
      
      sum = sum + outkenal(i,j);
    }
  }
  
  outkenal = outkenal/sum;
  
  return outkenal;
}



//image style 5: sketcher

//mirror index of index out of bounds
int mirrorIndex(int fetchI, int length){
  // code for mirror index at boundary
  if(fetchI < 0){
    return -fetchI;
  }
  else if(fetchI >= length){
    return 2*length - 2 - fetchI;
  }
  else{
    return fetchI;
  }
}

//image conv with kernel
arma::mat imageConv(arma::mat img,  arma::mat kernel){
  // code for image convolution
  int i,j,k,l,c,r,fetchI,fetchJ;
  double s;
  c = img.n_cols;
  r = img.n_rows;
  int ksize = kernel.n_cols/2;
    arma::mat changeimg(r,c);
  for(i = 0; i < r; i++){
    for(j = 0;j < c; j++){
      s = 0;
      for(k = -ksize; k < ksize+1; k++){
        for(l = -ksize; l < ksize+1; l++){
          fetchI = i+k;
          fetchJ = j+l;
          fetchI = mirrorIndex(fetchI,r);
          fetchJ = mirrorIndex(fetchJ,c);
          s += img(fetchI,fetchJ) * kernel(l+ksize,k+ksize);
        }
      }
      changeimg(i,j) = s;
    }
  }
  return changeimg;
}

//gauss filtter
arma::cube gaussfiltter(arma::cube img,int ksize, double sigma){
    arma::mat kernel = gaussfiltterkernel(ksize,sigma);
    arma::cube outimg(size(img));
  outimg.slice(0) = imageConv(img.slice(0),kernel);
  outimg.slice(1) = imageConv(img.slice(1),kernel);
  outimg.slice(2) = imageConv(img.slice(2),kernel);
  
  return outimg;
}

// image style 5: sketch
// [[Rcpp::export]]
arma::cube sketcher(arma::cube img, int ksize = 15,int cont = 5,double sigma = 50){
    arma::mat gray = img.slice(0);
    arma::mat dgray = 1-gray;
    arma::mat kernel = gaussfiltterkernel(ksize,sigma);
    for(int i =1;i<cont;i++){
        dgray = imageConv(dgray,kernel);
    }
    arma::mat outimg = dgray + gray;
  
    arma::cube sketch(size(img));
  sketch.slice(0) = outimg;
  sketch.slice(1) = outimg;
  sketch.slice(2) = outimg;
  
  return sketch;
}






