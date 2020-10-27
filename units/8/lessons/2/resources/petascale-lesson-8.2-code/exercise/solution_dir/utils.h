/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 8: OpenACC
 * Lesson 2: Intro to OpenACC
 * File: exercise/solution_dir/utils.h
 * Developed by Aaron Weeden for the Shodor Education Foundation, Inc.
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

// Checks if the return to malloc() is NULL, and exits if it is
void TryMalloc(void * const err)
{
  if (err == NULL)
  {
    fprintf(stderr, "malloc error\n");
    exit(EXIT_FAILURE);
  }
}

// Checks if the return to sprintf() is negative, and exits if it is
void TrySprintf(int const err)
{
  if (err < 0)
  {
    fprintf(stderr, "ERROR in sprintf\n");
    exit(EXIT_FAILURE);
  }
}

// Checks if the return to memcpy() is NULL, and exits if it is
void TryMemcpy(void * const err)
{
  if (err == NULL)
  {
    fprintf(stderr, "ERROR in memcpy\n");
    exit(EXIT_FAILURE);
  }
}

