typedef float realNumber;

//Defines Initials
#define RESOLUTION 256
#define IMAGE_HEIGHT RESOLUTION
#define IMAGE_WIDTH RESOLUTION
#define RAY_HEIGHT RESOLUTION
#define RAY_WIDTH RESOLUTION


struct pix_RGB{
    unsigned char R;
    unsigned char G;
    unsigned char B;
};


struct pix_realNumber{
    realNumber R;
    realNumber G;
    realNumber B;
};


pix_realNumber pixel_function(int width, int height, int maxWidth, int maxHeight);

void image_generate(pix_realNumber* pixels_in, pix_RGB* pixels_out);

void write_png_image(pix_RGB* pixels_all, char* filename);
