\name{Kaleido-package}
\alias{Kaleido-package}
\alias{Kaleido}
\docType{package}
\title{
\packageTitle{Kaleido}
}

\usage{
im_load(img)

feather(img)

fleetingtime(img)

lighting(img,centerx,centery,strength)

nostalgia(img)

sketcher(img,ksize,cont,sigma)

plot.nimg(img)

im_save(img, name, path, format, quality)
}
\arguments{
\item{img}{The path of the image to be processed.}

\item{centerx}{The abscissa of the location where the light source is placed.}

\item{centery}{The ordinate of the position where the light source is placed.}

\item{strength}{Light intensity, value range (0,1).}

\item{ksize}{Size of Gaussian filter kernel,15 for default.}

\item{cont}{Contrast of the sketch effect,default 5.}

\item{sigma}{The variance of Gaussian filter kernel,default 50.}

\item{name}{The name of the image when saved.}

\item{path}{The path of the image when saving.}

\item{format}{The image format you want to save as，PNG by defalut.}

\item{quality}{(jpg only) default is 0.95. Higher quality means less compression.Other format image don't need this parameter.}
}

\description{
\packageDescription{Kaleido}
}
\details{

In this package, you can use a function to read photos needing to be processed. It supports four image formats(png,bmp,jpg,jpeg) and provide 5 kinds of filter operation: Feather, fleetingtime, Nostalgia, Lighting and Sketcher. "Feather" can blur the edges of the picture and highlight the object or figure in the center."Fleetingtime" transforms the original image into a special effect with a sense of time and the precipitation of time, feeling the time slowly slipping away from the side. The "nostalgic" function turns a photo from the 20th century into an old photo from the 19th century, making the photo look full of age and vicissitudes."Lighting" provides similar the halo effect of lamplight, local increase brightness to an otherwise dark pictures. "Sketcher" function allows you to turn a photo into a sketch drawing.


\packageIndices{}
}
\author{
\packageAuthor{Kaleido}

Maintainer: \packageMaintainer{Kaleido}
}
\references{
Dirk Eddelbuettel and Romain Francois (2011). Rcpp: Seamless R and C++ Integration. Journal
of Statistical Software, 40(8), 1-18. URL http://www.jstatsoft.org/v40/i08/ and available as
vignette("Rcpp-introduction").
Eddelbuettel, Dirk (2013) Seamless R and C++ Integration with Rcpp. Springer, New York. ISBN
978-1-4614-6867-7. 

}
~~ Optionally other standard keywords, one per line, from file KEYWORDS in the R ~~
~~ documentation directory ~~
\keyword{ package }
\seealso{

 \code{\link[<pkg>:<pkg>-package]{<https://github.com/jeannie-ming-liu/Kaleido/tree/main/Kaleido>}} 
}
\examples{
library(Kaleido)
photo <- im_load("/Users/vera/Documents/somephoto/photo.jpg")
plot.nimg(photo)

# fether operation
photofet <- feather(photo)
plot.nimg(photofet)
im_save(photofet, "featherimg","/Users/vera/Documents/somephoto","jpg",0.85)

# Fleetingtime operation 
photofltme <- fleetingtime(photo)
plot.nimg(photofltme)
im_save(photofltme, "fleetingtimeimg","/Users/vera/Documents/somephoto","jpeg")

# Nostalgia operation
photonost <- fleetingtime(photo)
plot.nimg(photonost)
im_save(photonost, "nostalgiaimg","/Users/vera/Documents/somephoto","png")

# Lighting operation
photolight <- fleetingtime(photo)
plot.nimg(photolight)
im_save(photolight, "lightingimg","/Users/vera/Documents/somephoto")

# sketcher
photosketcher <- fleetingtime(photo)
plot.nimg(photosketcher)
im_save(photosketcher, "sketcherimg","/Users/vera/Documents/somephoto")

}