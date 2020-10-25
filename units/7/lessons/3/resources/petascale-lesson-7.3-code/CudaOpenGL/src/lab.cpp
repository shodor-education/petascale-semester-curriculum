// Support Code written by Michael D. Shah
// Please do not redistribute without asking permission.

// ==================== Libraries ==================
// Depending on the operating system we use
// The paths to SDL are actually different.
// The #define statement should be passed in
// when compiling using the -D argument.
// This gives an example of how a programmer
// may support multiple platforms with different
// dependencies.
#if defined(LINUX) || defined(MINGW)
	#include <SDL2/SDL.h>
#else
	// Windows and Mac use a different path
	// If you have compilation errors, change this as needed.
	#include <SDL.h>
#endif

#include "SDLWindow.h"

// The glad library helps setup OpenGL extensions.
#include <glad/glad.h>

// C and C++ Libraries
#include <stdio.h>
#include <string>
// ==================== Libraries ==================


// Our entry point into our program.
// Remember, C++ programs have exactly one entry point
// where the program starts.
int main(int argc, char* args[])
{
    SDLWindow win;
    win.loop();
	return 0;
}
