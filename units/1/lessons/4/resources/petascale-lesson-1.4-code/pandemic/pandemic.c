/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 1: OpenMP
 * Lesson 4: Scientific Examples on a Single Core
 * File: pandemic.c
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

#include <stdio.h>      // for printf
#include <stdlib.h>     // for malloc, free, and various others
#include <time.h>       // for time is used to seed the random number generator
#include <unistd.h>     // for random, getopt, some others
#ifdef X_DISPLAY
#include <X11/Xlib.h>   // for X display
#endif

#include "Defaults.h"
#include "Initialize.h"
#include "Infection.h"
#include "Core.h"
#include "Finalize.h"

#if defined(X_DISPLAY) || defined(TEXT_DISPLAY)
#include "Display.h"
#endif

int main(int argc, char ** argv) 
{
    /**** In Defaults.h ****/
    struct global_t global;
    struct const_t constant;
    struct stats_t stats;
    struct display_t dpy;
    /***********************/

    /***************** In Initialize.h *****************/
    init(&global, &constant, &stats, &dpy, &argc, &argv);
    /***************************************************/

    // Each process starts a loop to run the simulation 
    // for the specified number of days
    for(global.current_day = 0; global.current_day <= constant.total_number_of_days; 
        global.current_day++)
    {
        /****** In Infection.h ******/
        find_infected(&global);
        /****************************/

        /**************** In Display.h *****************/
        #if defined(X_DISPLAY) || defined(TEXT_DISPLAY)

        do_display(&global, &constant, &dpy);

        throttle(&constant);

        #endif
        /***********************************************/

        /************** In Core.h *************/
        move(&global, &constant);       

        susceptible(&global, &constant, &stats);

        infected(&global, &constant, &stats);

        update_days_infected(&global, &constant);
        /**************************************/
    }

    /******** In Finialize.h ********/
    show_results(&global, &stats);

    cleanup(&global, &constant, &dpy);
    /********************************/

    exit(EXIT_SUCCESS);
}
