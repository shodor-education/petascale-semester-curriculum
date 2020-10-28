/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 1: OpenMP
 * Lesson 4: Scientific Examples on a Single Core
 * File: Core.h
 * Developed by Aaron Weeden for the Shodor Education Foundation, Inc.
 * Modified by Yu Zhao, Macalester College (Modularized and restructured the
 * original code)
 *
 * Copyright (c) 2020 The Shodor Education Foundation, Inc.
 *
 * Browse and search the full curriculum at
 * <http://shodor.org/petascale/materials/semester-curriculum>.
 *
 * We welcome your improvements! You can submit your proposed changes to this
 * material and the rest of the curriculum in our GitHub repository at
 * <https://github.com/shodor-education/petascale-semester-curriculum>.
 *
 * We want to hear from you! Please let us know your experiences using this
 * material by sending email to petascale@shodor.org
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published
 * by the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 */

#ifndef PANDEMIC_DISPLAY_H
#define PANDEMIC_DISPLAY_H

#include <stdio.h>      // printf
#include <stdlib.h>     // malloc, free, and various others
#ifdef X_DISPLAY
#include <X11/Xlib.h>   // X display
#endif

void        init_display(struct const_t *constant, 
                struct display_t *dpy);
void        do_display(struct global_t *global,
                struct const_t *constant, struct display_t *dpy);
void        close_display(struct display_t *dpy);
void        throttle(struct const_t *constant);

/*
    init_display()
        Rank 0 initializes the graphics display
*/
void init_display(struct const_t *constant, struct display_t *dpy)
{
#ifdef X_DISPLAY
    /* Initialize the X Windows Environment
     * This all comes from 
     *   http://en.wikibooks.org/wiki/X_Window_Programming/XLib
     *   http://tronche.com/gui/x/xlib-tutorial
     *   http://user.xmission.com/~georgeps/documentation/tutorials/
     *      Xlib_Beginner.html
     */

    /* Open a connection to the X server */
    dpy->display = XOpenDisplay(NULL);
    if(dpy->display == NULL)
    {
        fprintf(stderr, "Error: could not open X dpy->display\n");
    }
    dpy->screen = DefaultScreen(dpy->display);
    dpy->window = XCreateSimpleWindow(dpy->display, RootWindow(dpy->display, dpy->screen),
        0, 0, constant->environment_width * PIXEL_WIDTH_PER_PERSON, 
        constant->environment_height * PIXEL_HEIGHT_PER_PERSON, 1,
        BlackPixel(dpy->display, dpy->screen), WhitePixel(dpy->display, dpy->screen));
    dpy->delete_window = XInternAtom(dpy->display, "WM_DELETE_WINDOW", 0);
    XSetWMProtocols(dpy->display, dpy->window, &dpy->delete_window, 1);
    XSelectInput(dpy->display, dpy->window, ExposureMask | KeyPressMask);
    XMapWindow(dpy->display, dpy->window);
    dpy->colormap = DefaultColormap(dpy->display, 0);
    dpy->gc = XCreateGC(dpy->display, dpy->window, 0, 0);
    XParseColor(dpy->display, dpy->colormap, dpy->red, &dpy->infected_color);
    XParseColor(dpy->display, dpy->colormap, dpy->green, &dpy->immune_color);
    XParseColor(dpy->display, dpy->colormap, dpy->white, &dpy->dead_color);
    XParseColor(dpy->display, dpy->colormap, dpy->black, &dpy->susceptible_color);
    XAllocColor(dpy->display, dpy->colormap, &dpy->infected_color);
    XAllocColor(dpy->display, dpy->colormap, &dpy->immune_color);
    XAllocColor(dpy->display, dpy->colormap, &dpy->susceptible_color);
    XAllocColor(dpy->display, dpy->colormap, &dpy->dead_color);
#endif
}

/*
    do_display()
        If display is enabled, Rank 0 displays a graphic of the current day
*/
void do_display(struct global_t *global, struct const_t *constant, struct display_t *dpy)
{
    #ifdef X_DISPLAY
    int current_person_id;

    char *states = global->states;
    int *x_locations = global->x_locations;
    int *y_locations = global->y_locations;

    XClearWindow(dpy->display, dpy->window);
    for(current_person_id = 0; current_person_id 
        <= global->number_of_people - 1; current_person_id++)
    {
        if(states[current_person_id] == INFECTED)
        {
            XSetForeground(dpy->display, dpy->gc, dpy->infected_color.pixel);
        }
        else if(states[current_person_id] == IMMUNE)
        {
            XSetForeground(dpy->display, dpy->gc, dpy->immune_color.pixel);
        }
        else if(states[current_person_id] == SUSCEPTIBLE)
        {
            XSetForeground(dpy->display, dpy->gc, dpy->susceptible_color.pixel);
        }
        else if(states[current_person_id] == DEAD)
        {
            XSetForeground(dpy->display, dpy->gc, dpy->dead_color.pixel);
        }
        else
        {
            fprintf(stderr, "ERROR: person %d has state '%c'\n",
                current_person_id, states[current_person_id]);
            exit(-1);
        }
        XFillRectangle(dpy->display, dpy->window, dpy->gc,
            x_locations[current_person_id] 
            * PIXEL_WIDTH_PER_PERSON, 
            y_locations[current_person_id]
            * PIXEL_HEIGHT_PER_PERSON, 
            PIXEL_WIDTH_PER_PERSON, 
            PIXEL_HEIGHT_PER_PERSON);
    }
    XFlush(dpy->display);
    #endif

    #ifdef TEXT_DISPLAY
    int current_person_id;
    int current_location_x;
    int current_location_y;
    int environment_height = constant->environment_height;
    int environment_width = constant->environment_width;

    char *states = global->states;
    int *x_locations = global->x_locations;
    int *y_locations = global->y_locations;

    for(current_location_y = 0; 
        current_location_y <= environment_height - 1;
        current_location_y++)
    {
        for(current_location_x = 0; current_location_x 
            <= environment_width - 1; current_location_x++)
        {
            dpy->environment[current_location_x][current_location_y] 
            = ' ';
        }
    }

    for(current_person_id = 0; 
        current_person_id <= global->number_of_people - 1;
        current_person_id++)
    {
        dpy->environment[x_locations[current_person_id]]
        [y_locations[current_person_id]] = 
        states[current_person_id];
    }

    printf("----------------------\n");
    for(current_location_y = 0;
        current_location_y <= environment_height - 1;
        current_location_y++)
    {
        for(current_location_x = 0; current_location_x 
            <= environment_width - 1; current_location_x++)
        {
            printf("%c", dpy->environment[current_location_x]
                [current_location_y]);
        }
        printf("\n");
    }
    #endif
}

/*
    close_display()
        If X display is enabled, then Rank 0 destroys the 
        X Window and closes the display
*/
void close_display(struct display_t *dpy)
{
#ifdef X_DISPLAY
    XDestroyWindow(dpy->display, dpy->window);
    XCloseDisplay(dpy->display);
#endif
}

/*
    throttle()
        Slows down the simulation to make X display easier to watch.
*/
void throttle(struct const_t *constant)
{
    // Wait between frames of animation
    usleep(constant->microseconds_per_day);
}

#endif
