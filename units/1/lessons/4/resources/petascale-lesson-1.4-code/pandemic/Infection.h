/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 1: OpenMP
 * Lesson 4: Scientific Examples on a Single Core
 * File: Infection.h
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

#ifndef PANDEMIC_INFECTION_H
#define PANDEMIC_INFECTION_H

void        find_infected(struct global_t *global);

/*
    find_infected()
        Each process determines its infected x locations 
        and infected y locations
*/
void find_infected(struct global_t *global)
{
    // counter to keep track of person in global struct
    int current_person_id;

    int current_infected_person = 0;
    for(current_person_id = 0; current_person_id <= global->number_of_people - 1; 
        current_person_id++)
    {
        if(global->states[current_person_id] == INFECTED)
        {
            global->infected_x_locations[current_infected_person] = 
            global->x_locations[current_person_id];
            global->infected_y_locations[current_infected_person] =
            global->y_locations[current_person_id];
            current_infected_person++;
        }
    }
}

#endif
