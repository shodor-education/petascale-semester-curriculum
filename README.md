<!--
*** Modified from othneildrew's README template
*** https://github.com/othneildrew/Best-README-Template/blob/master/README.md
-->


<!-- PROJECT LOGO -->
<br />
<p align="center">
 
<h3 align="center">Blue Waters Petascale Semester Course Curriculum</h3>
  <p align="center">
    The Blue Waters Petascale Semester Course Curriculum is a collection of 11 units and a total of 88 lessons that cover parallel computing and its applications. Units are groupings of lessons (about 6–12) with materials for each lesson designed to last 25 minutes of in-class time. The materials can be browsed and searched on <a href="http://shodor.org/petascale/materials/semesterCurriculum">Shodor's Petascale site.</a> 
    <br />
  </p>
</p>



<!-- TABLE OF CONTENTS -->
## Table of Contents

* [About the Project](#about-the-project)
* [Using the Materials](#using-the-materials)
* [Contributing](#contributing)
* [Curriculum Developers](#curriculum-developers)
* [License](#license)
* [Contact](#contact)



<!-- ABOUT THE PROJECT -->
## About The Project

<a href = "http://shodor.org">Shodor</a> is coordinating a curriculum development effort for which we are recruiting participants from institutions across the United States to fill a variety of roles (curriculum developers, testers, workshop instructors, and workshop participants). **Blue Waters Petascale Semester Course Curriculum** (or, simply, **"the Petascale Curriculum"**) aims to prepare faculty and professional staff to teach applied parallel modeling and simulation by incorporating the materials and lessons learned from the successful <a href="http://shodor.org/petascale/materials/institute/">Blue Waters Intern Petascale Institute</a> that has been conducted multiple times over the last ten years.

What has set the Petascale Institute apart from other efforts is that it starts with <a href="http://shodor.org/petascale/materials/modules"> working models solving problems in real science</a> that cover the continuum of multi-core/many-core technologies as exemplified by Blue Waters (OpenMP, MPI, OpenACC, CUDA, hybrid). Instead of teaching these from an abstract, theoretical point, we have exploited the basic motifs used in real scientific code to teach undergraduate and graduate students the basics of applied parallel modeling and simulation.

The goal of the Petascale Curriculum effort is to update the Petascale Institute materials to become more useful for preparing undergraduate and graduate students, as well as to help faculty and professional staff to integrate the materials into their own courses, student programs, REUs, workshops, and institutes.

This project is a component of the Blue Waters education initiatives at the University of Illinois, funded by the National Science Foundation.


<!-- HOW TO USE -->
## Using the Materials

These resources can be helpful both for *instructors* looking to integrate parts of parallel computing material into their course or in developing their own parallel computing course and for *students* looking to learn more about parallel computing in general. The curriculum materials can be accesed on [Shodor's Petascale site](http://shodor.org/petascale/materials/semesterCurriculum/units/).

For instructors:

* These materials are meant to provide example lessons that you can incorporate into your own course. We designed each lesson to take ~25 minutes of in class time so you can easily fit two topics into a 50 minute or 75 minute lecture.
* Additionally, Shodor plans to host a semesterly webinar for instructors looking to develop their parallel computing course. The webinars will be in August, September and December and more information on these webinars is TBD. If you are interested, please fill out the contact form at the bottom of any lesson page on the website.

For students:

* Watch our videos example lectures! Our developers put a lot of time into making these to help you understand the concepts more thoroughly.
* Go through our example code and example lesson plans/lecture notes.


<!-- CONTRIBUTING -->
## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any contributions you make are **greatly appreciated**.


1. Fork the Project
2. Clone it to your local system
3. Make a new branch
4. Using the tables below, determine if your file is to be uploaded as a resource (indicated in the file type column)

**Unit Page**
<table>
  <tr>
   <td><strong>Material</strong>
   </td>
   <td><strong>File Type</strong>
   </td>
   <td><strong>File Name</strong>
   </td>
  </tr>
  <tr>
   <td>Unit Learning Objectives
   </td>
   <td>HTML
   </td>
   <td>"objectives.html"
   </td>
  </tr>
  <tr>
   <td>Unit Student Prerequisites
   </td>
   <td>HTML
   </td>
   <td>"studentPrereqs.html"
   </td>
  </tr>
  <tr>
   <td>Unit Instructor Prerequisites
   </td>
   <td>HTML
   </td>
   <td>"instructorPrereqs.html"
   </td>
  </tr>
  <tr>
   <td>Unit Sample Assessment
   </td>
   <td>Resource - PDF/DOCX
   </td>
   <td>"petascale-unit-#-assessment.pdf" and "petascale-unit-#-assessment.docx"
   </td>
  </tr>
  <tr>
   <td>Unit Title/Description
   </td>
   <td>JSON
   </td>
   <td>"pageDetails.json"
   </td>
  </tr>
</table>


**Lesson Page**
<table>
  <tr>
   <td><strong>Material</strong>
   </td>
   <td><strong>File Type</strong>
   </td>
   <td><strong>File Name / Extension</strong>
   </td>
  </tr>
  <tr>
   <td>Lesson Learning Objectives
   </td>
   <td>HTML
   </td>
   <td>"objectives.html"
   </td>
  </tr>
  <tr>
   <td>Lesson Student Prerequisites
   </td>
   <td>HTML
   </td>
   <td>"studentPrereqs.html"
   </td>
  </tr>
  <tr>
   <td>Lesson Instructor Prerequisites
   </td>
   <td>HTML
   </td>
   <td>"instructorPrereqs.html"
   </td>
  </tr>
  <tr>
   <td>Lesson Sample Assessment
   </td>
   <td>Resource - PDF/DOCX
   </td>
   <td>"petascale-lesson-#.#-assessment.pdf" and "petascale-lesson-#.#-assessment.docx"
   </td>
  </tr>
  <tr>
  <tr>
   <td>Presentation Slides
   </td>
   <td>Resource - PDF/PPTX
   </td>
   <td>"petascale-lesson-#.#slides.pdf" and "petascale-lesson-#.#slides.pptx"
   </td>
  </tr>
  <tr>
   <td>Presentation Video
   </td>
   <td>HTML (link to video)
   </td>
   <td>"presentationVideo.html"
   </td>
  </tr>
  <tr>
   <td>Instructor Guide
   </td>
   <td>Resource - PDF/DOCX
   </td>
   <td>"petascale-lesson-#.#-instructorGuide.pdf" and "petascale-lesson-#.#-instructorGuide.docx"
   </td>
  </tr>
  <tr>
   <td>Code Example
   </td>
   <td>Resource - GNU zip
   </td>
   <td>"petascale-lesson-#.#-code.tar.gz"
   </td>
  </tr>
  <tr>
   <td>Materials Needed and System Requirements
   </td>
   <td>HTML
   </td>
   <td>"systemReqs.html"
   </td>
  </tr>
  <tr>
   <td>Exercise Instructions for Students
   </td>
   <td>Resource - PDF/DOCX
   </td>
   <td>"petascale-lesson-#.#-exercise.pdf" and "petascale-lesson-#.#-exercise.docx"
   </td>
  </tr>
  <tr>
  <tr>
   <td>Curriculum Standards Addressed
   </td>
   <td>HTML
   </td>
   <td>"currStandards.html"
   </td>
  </tr>
   <td>References and Further Reading
   </td>
   <td>Resource - PDF/DOCX
   </td>
   <td>"petascale-lesson-#.#-references.pdf" and "petascale-lesson-#.#-references.docx"
   </td>
  </tr>
  <tr>
   <td>Lesson Title/Description/Author
   </td>
   <td>JSON
   </td>
   <td>"pageDetails.json"
   </td>
  </tr>
</table>

* If you would like to update the description for a unit or lesson, simply edit the description line in the appropriate .json file

* If the file type *is not* preceded by "resource" in the table, you will have to upload your materials by editing the existing version of the HTML file that is in your branch. Make sure that the file you are editing is in the correct folder and is named correctly according to the table
    * You may also make changes to the curriculum material using your own text editor instead of directly editing the HTML. You can use a tool such as [Word 2 Clean HTML](https://word2cleanhtml.com/) for Word documents or [Docs to Markdown](https://gsuite.google.com/marketplace/app/docs_to_markdown/700168918607?pann=cwsdp&hl=en-US) for Google documents to convert text into HTML

* If the file type *is* preceded by "resource" in the table, please name your materials using the naming scheme provided and then upload your materials to the resources folder in the appropriate unit or lesson. This extra step allows the resources to be displayed directly on the webpage for each unit and lesson
5. Commit your changes
6. Click the Compare & pull request button
7. Click Create pull request to open a new pull request, and provide necessary details on what you’ve changed



<!-- CURRICULUM DEVELOPERS -->
## Curriculum Developers
* Colleen Heinemann, University of Illinois at Urbana-Champaign
* David P. Bunde, Knox College
* David A. Joiner, Kean University
* Hyacinthe Aboudja, Oklahoma City University
* Justin Oelgoetz, Austin Peay State University
* Marc Gagné, West Chester University
* Maria Pantoja, CalPoly San Luis Obispo
* Michael N Groves, California State University Fullerton
* Michael D. Shah, Northeastern University
* Nitin Sukhija, Slippery Rock University of Pennsylvania
* Paul F. Hemler, Hampden-Sydney College
* Peter J. Hawrylak, The University of Tulsa
* Roman Voronov, New Jersey Institute of Technology
* Sanish Rai, West Virginia University Institute of Technology
* Widodo Samyono, Jarvis Christian College

<!-- LICENSE -->
## License
License pending
<!-- Distributed under the MIT License. See `LICENSE` for more information. -->



<!-- CONTACT -->
## Contact

Via the Web
* Give us feedback in our [feedback form](http://shodor.org/petascale/feedback/).
* Please [contact us](http://shodor.org/petascale/contact/) if you have any questions, comments, or concerns.

Via Mail or Phone:
* The Shodor Education Foundation, Inc.
* 701 William Vickers Ave
     Durham, NC 27701
* Tel: (919) 530-1911
* Fax: (919) 530-1944


Project Link: [https://github.com/shodor-education/petascale-semester-curriculum](https://github.com/shodor-education/petascale-semester-curriculum)
