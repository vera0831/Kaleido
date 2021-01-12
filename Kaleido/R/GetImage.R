#' Image Loading
#' The [im_load()] function loads image from a local path.
#' @title Kaleido
#' @import dplyr
#' @import readbitmap
##' @param file refers to the path of the file
##' @return an image on r plot
##' @author Haitao Zhao,Wanyun Lu,Jiaming Liu
##' @export

library(dplyr)
library(readbitmap)


##input image
im_load = function( file){
  ext = sub( ".*\\.([^.]{3,4})$", "\\1", file ) %>% tolower
  if( ext %in% c( "png", "bmp", "jpg", "jpeg" ) ){
    tryCatch({
      im = readbitmap::read.bitmap( file )
    },
    error = function(e) {
      stop( paste0( e, "Note: im_load() fails for binary (black/white) bmp image." ) )
    })
    if( length( dim( im ) ) == 2 ){ # gray-scale image
      dim( im ) = c( dim( im ), 1 )
    } else if( length( dim( im ) ) == 3 ){ # multiple channels
      if( dim( im )[ 3 ] %in% c( 2, 4 ) ){
        # remove alpha channel if it is uninformative
        if( min( im[ , , dim( im )[ 3 ] ] ) == max( im[ , , dim( im )[ 3 ] ] ) ){
          im = im[ , , 1:( dim( im )[ 3 ] - 1 ), drop = FALSE ]
        }
      }
    }
    return( im )
  } else {
    stop( "Only jpg, png, and bmp formats are supported." )
  }
}




## save image
im_save = function( im, name, path, format = "png", quality = .95 ){
  if( ! format %in% c( "jpg", "png","jpeg" ) ){
    warning( "Incorrect imaeg format. Use either jpg or png." )
    return()
  }
  if( base::missing( name ) ){
    name = deparse( substitute( im ) )
  }
  if( im_nc( im ) == 1 ){
    im = im_rep( im )
  }
  if( stringr::str_sub( path, stringr::str_length( path ) ) == "/" ){
    path = stringr::str_sub( path, end = stringr::str_length( path ) - 1 )
  }
  if( max( im ) > 1 || min( im ) < 0 ){
    # warning( "Pixcel value exceeds the range [0,1], and hence it was clamped when saving.")
    im = clamping( im )
  }
  base::dir.create( path, showWarnings = FALSE, recursive = TRUE )
  file = paste0( path, "/", name, ".", format )
  if( format == "png" ){
    png::writePNG( im, file )
  } else if ( format == "jpeg" | format == "jpg" ){
    jpeg::writeJPEG( im, file, quality = quality )
  }
}


##plot image
plot.nimg = function( x, ... ){
  old.par = graphics::par( no.readonly = TRUE )
  on.exit( graphics::par( old.par ), add = TRUE )
  if( im_npix( x ) == 0 ){
    stop( "The image is empty." )
  }
  
  if( im_nc( x ) == 1 ){
    # a raster array must have exactly 3 or 4 planes
    x = im_rep( x)
  }
  im = x[ ,,, drop = FALSE ]
  if( max( im ) > 1 || min( im ) < 0 ){
    im = clamping( im )
  }
  graphics::par( mar = c( 0, 0, 0, 0 ) )
  graphics::plot.new()
  graphics::plot.window(
    xlim = c(1,im_width(x)), ylim = c(im_height(x),1), asp = 1, xaxs = "i", yaxs = "i", ...)
  rst = grDevices::as.raster(
    im, rescale = FALSE, colorscale = NULL, colourscale = NULL, col.na = grDevices::rgb(0,0,0,0) )
  graphics::rasterImage( rst, 1, nrow( rst ), ncol( rst ), 1, interpolate = FALSE )
  invisible( x )
}
