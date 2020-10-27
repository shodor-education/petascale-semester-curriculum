/* Blue Waters Petascale Semester Curriculum v1.0
 * Unit 8: OpenACC
 * Lesson 2: Intro to OpenACC
 * File: exercise/solution_dir/heat-openacc.c
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

/* Model of heat diffusion - a 2D rectangular environment has its left edge with
   a fixed non-zero heat; the other 3 edges have fixed zero heat. The middle of 
   the environment starts with zero heat, but as the model advances, the heat
   of each cell in the middle gets set to the average of its 4 neighbors. The
   model advances until the amount of overall heat change from one time step to
   the next is sufficiently small.
 */

/*******************************************************************************
  IMPORT LIBRARIES
 ******************************************************************************/
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "utils.h"

/*******************************************************************************
  DEFINE MACROS
 ******************************************************************************/
#define ROW_COUNT 21
#define COLUMN_COUNT 25
#define INIT_LEFT_HEAT 100.0 // The left edge of the environment has this
                             // fixed heat
#define MIN_HEAT_DIFF 10 // If the overall system heat changes by less
                         // than this amount between two time steps, the
                         // model stops
#define OUTPUT_HEAT_LEN 6 // Number of characters needed to print each heat
                          // value
#define OUTPUT_DIGS_AFTER_DEC_PT 2  // Number of digits to print after the
                                    // decimal point for each heat value

// Convert index within NewHeats array (which only includes the middle cells)
// into index within Heats array (which also includes the edge cells)
#define NEW_TO_OLD(idx) ((idx) + (COLUMN_COUNT) + 1 + \
                         2 * ((idx) / ((COLUMN_COUNT)-2)))

// Convert index within Heats array into index within OutputStr string
#define OUTPUT_IDX(idx) ((idx) * ((OUTPUT_HEAT_LEN)+1) + \
                         (idx) / (COLUMN_COUNT))

/*******************************************************************************
  DECLARE GLOBAL VARIABLES
 ******************************************************************************/
int CellCount; // Total number of cells in the environment
int CellCountWithoutEdges; // Total number of cells in the environment, not
                           // counting the edges
int CellFloatByteCount; // Total number of bytes if there are enough floats for
                        // each cell
int CellFloatByteCountWithoutEdges; // Total number of bytes if there are
                                    // enough floats for each cell, not
                                    // counting the edges
int CellCharByteCount; // Total number of bytes if there are enough chars for
                       // each cell
float * Heats; // Array of heat values for each cell
float * NewHeats; // Array of heat values for each cell in the next time step
float * Diffs; // Array of differences between the heat values for each cell
               // in the current and next time steps
char * OutputStr; // String to output at each time step
bool IsStillRunning; // Used to keep track of whether the model should continue
                     // into the next time step
int TimeIdx; // The current time step

/*******************************************************************************
  DECLARE FUNCTIONS
 ******************************************************************************/
void Init();
  void InitCellCounts();
  void InitMemory();
  void InitHeats();
    void SetOutputStr(int const cellIdx);
  void InitLeft();
  void InitOutputStrNewlines();
void Simulate();
  void SetNewHeats();
    void AverageNeighborHeats();
  void PrintOutputStr();
  void AdvanceHeats();
void Finalize();

/*******************************************************************************
  DEFINE FUNCTIONS
 ******************************************************************************/
int main()
{
  Init();

  #pragma acc data copyin(Heats[0:CellCount], \
                       NewHeats[0:CellCountWithoutEdges]) \
                   create(Diffs[0:CellCountWithoutEdges])
  {
    Simulate();
  }

  Finalize();
  return 0;
}

// Preconditions: none
// Postconditions: Heats has been initialized
//                 OutputStr has been initialized
void Init()
{
  InitCellCounts();
  InitMemory();
  InitHeats();
  InitLeft();
  InitOutputStrNewlines();
}

// Preconditions: none
// Postconditions: CellCount has been defined
//                 CellCountWithoutEdges has been defined
//                 CellFloatByteCount has been defined
//                 CellFloatByteCountWithoutEdges has been defined
//                 CellCharByteCount has been defined
void InitCellCounts()
{
  CellCount = ROW_COUNT * COLUMN_COUNT;
  CellCountWithoutEdges = CellCount - 2 * (ROW_COUNT + COLUMN_COUNT - 2);
  CellFloatByteCount = CellCount * sizeof(float);
  CellFloatByteCountWithoutEdges = CellCountWithoutEdges * sizeof(float);
  CellCharByteCount = CellCount * sizeof(char);
}

// Preconditions: CellFloatByteCount has been defined
//                CellFloatByteCountWithoutEdges has been defined
//                CellCharByteCount has been defined
// Postconditions: Memory has been allocated for Heats
//                 Memory has been allocated for NewHeats
//                 Memory has been allocated for Diffs
//                 Memory has been allocated for OutputStr
void InitMemory()
{
  TryMalloc(Heats     = (float*)malloc(CellFloatByteCount));
  TryMalloc(NewHeats  = (float*)malloc(CellFloatByteCountWithoutEdges));
  TryMalloc(Diffs     = (float*)malloc(CellFloatByteCountWithoutEdges));
  TryMalloc(OutputStr = (char*) malloc(CellCharByteCount *
                                       (OUTPUT_HEAT_LEN+1) +
                                       ROW_COUNT * sizeof(char)));
}

// Preconditons: Memory has been allocated for Heats
// Postconditions: Heats has been initialized, except for the left edge
void InitHeats()
{
  for (int cellIdx = 0; cellIdx < CellCount; cellIdx++)
  {
    Heats[cellIdx] = 0.0;
    SetOutputStr(cellIdx);
  }
}

// Preconditons: Heats has been initialized, except for the left edge
// Postconditions: Heats has been initialized
void InitLeft()
{
  for (int rowIdx = 0; rowIdx < ROW_COUNT; rowIdx++)
  {
    int const cellIdx = rowIdx * COLUMN_COUNT;
    Heats[cellIdx] = INIT_LEFT_HEAT;
    SetOutputStr(cellIdx);
  }
}

// Preconditons: Memory has been allocated for OutputStr
// Postconditions: OutputStr has been initialized
void InitOutputStrNewlines()
{
  for (int rowIdx = 0; rowIdx < ROW_COUNT; rowIdx++)
  {
    OutputStr[OUTPUT_IDX((rowIdx+1) * (COLUMN_COUNT)) - 1] = '\n';
  }
}

// Preconditions: Heats has not been updated at TimeIdx and cellIdx
// Postconditions: OutputStr has been updated at TimeIdx and cellIdx
void SetOutputStr(int const cellIdx)
{
  char tmp[OUTPUT_HEAT_LEN+2];
  TrySprintf(sprintf(tmp,
                     "%*.*f ",
                     OUTPUT_HEAT_LEN,
                     OUTPUT_DIGS_AFTER_DEC_PT,
                     Heats[cellIdx]));
  TryMemcpy(memcpy(&(OutputStr[OUTPUT_IDX(cellIdx)]),
                   tmp,
                   OUTPUT_HEAT_LEN+1));
}

// Preconditons: Heats has been initialized
//               OutputStr has been initialized
// Postconditions: The simulation has run
void Simulate()
{
  IsStillRunning = true;
  for (TimeIdx = 0; IsStillRunning; TimeIdx++)
  {
    SetNewHeats();
    PrintOutputStr();

    #pragma acc update device(NewHeats[0:CellCountWithoutEdges])

    AdvanceHeats();

    #pragma acc update host(Heats[0:CellCount])
  }
}

// Preconditions: Heats has not been updated at TimeIdx
// Postconditions: OutputStr has been updated at TimeIdx
//                 NewHeats has been updated at TimeIdx
//                 IsStillRunning has possibly been updated
void SetNewHeats()
{
  for (int cellIdx = 0; cellIdx < CellCountWithoutEdges; cellIdx++)
  {
    SetOutputStr(NEW_TO_OLD(cellIdx));
  }

  #pragma acc update device(Heats[0:CellCount])

  AverageNeighborHeats();

  #pragma acc update host(NewHeats[0:CellCountWithoutEdges], \
                             Diffs[0:CellCountWithoutEdges])

  // Prepare to stop the simulation if the heat is not changing much
  float totalDiff = 0.0;
  for (int cellIdx = 0; cellIdx < CellCountWithoutEdges; cellIdx++)
  {
    totalDiff += Diffs[cellIdx];
  }
  if (totalDiff < MIN_HEAT_DIFF)
  {
    IsStillRunning = false;
  }
}

// Preconditions: Heats has not been updated at TimeIdx and cellIdx
// Postconditions: NewHeats has been updated at TimeIdx and cellIdx
//                 Diffs has been updated at TimeIdx and cellIdx
void AverageNeighborHeats()
{
  #pragma acc parallel loop present(Heats[0:CellCount], \
                                 NewHeats[0:CellCountWithoutEdges], \
                                    Diffs[0:CellCountWithoutEdges])
  for (int cellIdx = 0; cellIdx < CellCountWithoutEdges; cellIdx++)
  {
    NewHeats[cellIdx] = 0.25 * (Heats[NEW_TO_OLD(cellIdx) - COLUMN_COUNT] +
                                Heats[NEW_TO_OLD(cellIdx) - 1] +
                                Heats[NEW_TO_OLD(cellIdx) + 1] +
                                Heats[NEW_TO_OLD(cellIdx) + COLUMN_COUNT]);

    Diffs[cellIdx] = NewHeats[cellIdx] - Heats[NEW_TO_OLD(cellIdx)];
  }
}

// Preconditions: OutputStr has been updated at TimeIdx
// Postconditions: OutputStr has been printed at TimeIdx
void PrintOutputStr()
{
  printf("Time step %d\n%s\n", TimeIdx, OutputStr);
}

// Preconditions: NewHeats has been updated at TimeIdx
// Postconditions: Heats has been updated at TimeIdx
void AdvanceHeats()
{
  #pragma acc parallel loop present(Heats[0:CellCount], \
                                 NewHeats[0:CellCountWithoutEdges])
  for (int cellIdx = 0; cellIdx < CellCountWithoutEdges; cellIdx++)
  {
    Heats[NEW_TO_OLD(cellIdx)] = NewHeats[cellIdx];
  }
}

// Preconditions: The simulation has run
// Postconditions: Memory has been freed for Heats
//                 Memory has been freed for NewHeats
//                 Memory has been freed for Diffs
//                 Memory has been freed for OutputStr
void Finalize()
{
  free(OutputStr);
  free(Diffs);
  free(NewHeats);
  free(Heats);
}

