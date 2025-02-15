#include "kernel.cuh"

// function to generate image with anti-alias
__host__ void image_generate_smooth(pix_realNumber* pixels_in, pix_RGB* pixels_out, config* cfgs) 
{
    realNumber color_temp1, color_temp2, color_temp3;

    for (int y = 0;y < cfgs->img_h;++y)
    {
        for (int x = 0;x < cfgs->img_w;++x)
        {
            color_temp1 = 0.0;
            color_temp2 = 0.0;
            color_temp3 = 0.0;
            
            // for every 4 pixels in a square
            for (int y2 = 0;y2 < 4;++y2)
            {
                for (int x2 = 0;x2 < 4;++x2)
                {
                    // produce one single pixel
                    color_temp1 += std::min(std::max(pixels_in[x * 4 + x2 + (y * 4 + y2) * cfgs->ray_w].R, 0.0f), 1.0f);
                    color_temp2 += std::min(std::max(pixels_in[x * 4 + x2 + (y * 4 + y2) * cfgs->ray_w].G, 0.0f), 1.0f);
                    color_temp3 += std::min(std::max(pixels_in[x * 4 + x2 + (y * 4 + y2) * cfgs->ray_w].B, 0.0f), 1.0f);
                }
            }
            pixels_out[x + y * cfgs->img_w].R = static_cast<unsigned char>(255.0f * color_temp1 / 16.0);
            pixels_out[x + y * cfgs->img_w].G = static_cast<unsigned char>(255.0f * color_temp2 / 16.0);
            pixels_out[x + y * cfgs->img_w].B = static_cast<unsigned char>(255.0f * color_temp3 / 16.0);
        }
    }

}

// utility function to produce PNG
__host__ void write_png_image(pix_RGB* pixels_all, char* filename, config* cfgs)
{
    png_byte** row_pointers; // pointer to image bytes
    FILE* fp; // file for image

    do // one time do-while to properly free memory and close file after error - why??
    {
        row_pointers = (png_byte**)malloc(sizeof(png_byte*) * cfgs->img_h);

        if (!row_pointers)
        {
            printf("Allocation failed\n");
            break;
        }
        for (int i = 0; i < cfgs->img_h; i++)
        {
            row_pointers[i] = (png_byte*)malloc(4 * cfgs->img_w);
            if (!row_pointers[i])
            {
                printf("Allocation failed\n");
                break;
            }
        }
        // fill image with color
        for (int y = 0; y < cfgs->img_h; y++)
        {
            for (int x = 0; x < cfgs->img_w; ++x)
            {
                row_pointers[y][x * 4] = pixels_all[x + y * cfgs->img_w].R;
                row_pointers[y][x * 4 + 1] = pixels_all[x + y * cfgs->img_w].G;
                row_pointers[y][x * 4 + 2] = pixels_all[x + y * cfgs->img_w].B;
                row_pointers[y][x * 4 + 3] = 255; //Alpha
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
            cfgs->img_w, //image width
            cfgs->img_h, //image height
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
    } while (0);
    //close file
    if (fp)
    {
        fclose(fp);
    }
    //free allocated memory
    for (int i = 0; i < cfgs->img_h; i++)
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
