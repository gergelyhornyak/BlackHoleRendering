#include "Functions.h"
#include "./eigen-3.4.0/Eigen/Dense"
#include <png.h>

//Main Pixel Shader
pix_realNumber pixel_function(int width, int height, int maxWidth, int maxHeight)
{
    pix_realNumber pixel;

    //Define UV Coordinates
    realNumber u = (realNumber(width) / realNumber(maxWidth)) - 0.5f;
    realNumber v = (realNumber(maxHeight-height) / realNumber(maxHeight)) - 0.5f;

    //Define ray
    realNumber FocalLength = 6.0f;
    Eigen::Vector3d fragmentCoords(u * 0.01f,v * 0.01f,0.0f);
    Eigen::Vector3d camPos(0.0f,5.0f,110.0f);
    Eigen::Vector3d rayOrig(0.0f,0.0f,(FocalLength/1000.0f));
    Eigen::Vector3d rayDir(0.0f,0.0f,0.0f);
    rayOrig += camPos;
    rayDir = (fragmentCoords + camPos) - rayOrig;
    rayDir.normalize();

    //Define Initials
    realNumber dx = 0.5f;
    realNumber r = 0.0f;
    realNumber G = 1.0f;
    realNumber M = 1.0f;
    Eigen::Vector3d Temp(0.0f,0.0f,0.0f);
    Eigen::Vector3d dir(0.0f,0.0f,0.0f);
    realNumber mag = 0.0f;

    //sqrt(rayOrig[0]*rayOrig[0] + rayOrig[1]*rayOrig[1] + rayOrig[2]*rayOrig[2])

    pixel.R = 0;
    pixel.G = 0;
    pixel.B = 0;

    int i=0;

    //March
    for(i = 0; i < 1700; i++)
    {
        r = sqrt(rayOrig[0]*rayOrig[0] + rayOrig[1]*rayOrig[1] + rayOrig[2]*rayOrig[2]);
        dx = 0.1;
        
        //Event Horizon Case
        if(r < 2.0f*G*M)
        {
            break;
        }
        
        if(r > 120)
        {
            break;
        }

        if(r > 10.0f && r < 70 && abs(rayOrig[1]) < 0.2f)
        {
            pixel.R = 1.0f;
            pixel.G = 1.0f;
            pixel.B = 1.0f;
            break;
        }

        //Update Ray
        Temp = rayOrig;
        Temp.normalize();
        mag = (-2.0f*G*M) / (r*r);
        Temp *= mag * 0.2f;
        dir = rayDir + Temp;
        dir.normalize();
        rayDir = dir;
        rayOrig += rayDir*dx;
    }
    return pixel;
}

void image_generate(pix_realNumber* pixels_in, pix_RGB* pixels_out){

    realNumber color_temp;
    for(int y=0;y<IMAGE_HEIGHT;++y)
    {
        for(int x=0;x<IMAGE_WIDTH;++x)
        {
            color_temp=std::min(std::max(pixels_in[x+y*RAY_WIDTH].R,0.0f),1.0f);
            pixels_out[x+y*IMAGE_WIDTH].R=static_cast<unsigned char>(255.0f*color_temp);
            color_temp=std::min(std::max(pixels_in[x+y*RAY_WIDTH].G,0.0f),1.0f);
            pixels_out[x+y*IMAGE_WIDTH].G=static_cast<unsigned char>(255.0f*color_temp);
            color_temp=std::min(std::max(pixels_in[x+y*RAY_WIDTH].B,0.0f),1.0f);
            pixels_out[x+y*IMAGE_WIDTH].B=static_cast<unsigned char>(255.0f*color_temp);
        }
    }

}



void write_png_image(pix_RGB* pixels_all, char* filename)
{
    png_byte** row_pointers; // pointer to image bytes
    FILE* fp; // file for image

    do // one time do-while to properly free memory and close file after error
    {
        row_pointers = (png_byte**)malloc(sizeof(png_byte*) * IMAGE_HEIGHT);
        if (!row_pointers)
        {
            printf("Allocation failed\n");
            break;
        }
        for (int i = 0; i < IMAGE_HEIGHT; i++)
        {
            row_pointers[i] = (png_byte*)malloc(4*IMAGE_WIDTH);
            if (!row_pointers[i])
            {
                printf("Allocation failed\n");
                break;
            }
        }
        // fill image with color
        for (int y = 0; y < IMAGE_HEIGHT; y++)
        {
            for (int x = 0; x < IMAGE_WIDTH; ++x)
            {
                row_pointers[y][x*4] = pixels_all[x+y*IMAGE_WIDTH].R;
                row_pointers[y][x*4 + 1] = pixels_all[x+y*IMAGE_WIDTH].G;
                row_pointers[y][x*4 + 2] = pixels_all[x+y*IMAGE_WIDTH].B;
                row_pointers[y][x*4 + 3] = 255; //Alpha
            }
        }
        //printf("%d %d %d %d\n", row_pointers[0][0], row_pointers[0][1], row_pointers[0][2], row_pointers[0][3]);

        fp = fopen(filename, "wb"); //create file for output
        if (!fp)
        {
            printf("Open file failed\n");
            break;
        }
        png_struct* png = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL); //create structure for write
        if (!png)
        {
            printf("Create write struct failed\n");
            break;
        }
        png_infop info = png_create_info_struct(png); // create info structure
        if (!info)
        {
            printf("Create info struct failed\n");
            break;
        }
        if (setjmp(png_jmpbuf(png))) // this is some routine for errors?
        {
            printf("setjmp failed\n");
        }
        png_init_io(png, fp); //initialize file output
        png_set_IHDR( //set image properties
            png, //pointer to png_struct
            info, //pointer to info_struct
            IMAGE_WIDTH, //image width
            IMAGE_HEIGHT, //image height
            8, //color depth
            PNG_COLOR_TYPE_RGBA, //color type
            PNG_INTERLACE_NONE, //interlace type
            PNG_COMPRESSION_TYPE_DEFAULT, //compression type
            PNG_FILTER_TYPE_DEFAULT //filter type
            );
        png_write_info(png, info); //write png image information to file
        png_write_image(png, row_pointers); //the thing we gathered here for
        png_write_end(png, NULL);
        printf("Image was created successfully\nCheck %s file\n", filename);
    } while(0);
    //close file
    if (fp)
    {
        fclose(fp);
    }
    //free allocated memory
    for (int i = 0; i < IMAGE_HEIGHT; i++)
    {
        if (row_pointers[i])
        {
            free(row_pointers[i]);
        }
    }
    if (row_pointers)
    {
        free(row_pointers);
    }
}
