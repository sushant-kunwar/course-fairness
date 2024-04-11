from gurobipy import *
from functools import reduce
import csv
import re
import json


model = Model()
obj = LinExpr()

credit_cap = 4

priority_1 = 8
priority_2 = 4
priority_3 = 2
priorities = {"CS PhD": priority_1, "CS MS": priority_1, "TSB PhD": priority_1, "CSLS PhD": priority_1, "CE PhD": priority_1, 
              "CE MS": priority_2, "MSR MS": priority_2, "MSR MSR": priority_2, "Robotics	MS": priority_2, "Robotics	MSR": priority_2,
              "MSIT": priority_3, "EE MS": priority_3, "EE PhD": priority_3, "MSIT MSIT": priority_3}

student_to_courses_dict = {}
student_to_vars_dict = {}
student_to_csv_rows_dict = {}
student_to_major_dict = {}
student_to_degree_dict = {}
course_to_students_dict = {}
course_to_MS_students_dict = {}
course_capacities = {}
dummy_vars = []

course_stats = {}

def ParseCamelCase(string):
  return re.sub('([A-Z][a-z]*)', r' \1', re.sub('([A-Z]+)', r' \1', string)).split()

def parse_course_time(course_name):
  weekday_to_number_dict = {"M": 0, "Tu": 1, "W": 2, "Th": 3, "F": 4} # don't have Saturday and Sunday
  words_list = course_name.split(" ")
  beginning_time_number = re.findall("[0-9]+:[0-9]+", course_name)
  beginning_and_end_times = re.findall("[0-9]+:[0-9]+ [A|P]M", course_name)
  
  # converting Weekday hh:mm XM-hh:mm XM to integers
  for i in range(2):
    time = beginning_and_end_times[i]
    if time[-2:] == "AM" or (time[-2:] == "PM" and time[:2] == "12"):
      beginning_and_end_times[i] = int(time[:2]) * 60 + int(time[3:5])
    else:
      beginning_and_end_times[i] = 720 + int(time[:2]) * 60 + int(time[3:5])

  time_index =  words_list.index(beginning_time_number[0])
  days_of_week = ParseCamelCase(words_list[time_index-1])
  time_ranges = [range(weekday_to_number_dict[x]*1440 + beginning_and_end_times[0], weekday_to_number_dict[x]*1440 + + beginning_and_end_times[1]) for x in days_of_week]
  union = reduce((lambda x, y: x.union(y)), time_ranges, set())
  return union

def must_have_x_of_top_y_courses(x, y, student_vars):
  model.addConstr(sum(student_vars[:y]) >= min(len(student_vars), x))

with open('win22-course-data.csv', newline='') as csvfile:
  y = 4 # top y of student preferences
  courses_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
  for row in courses_reader:
    if (row[0] == "SIS Number"): # skip header row
      continue
    course = row[5]
    course_capacities[course] = int(row[2][:-1])/100 * int(row[1])
    

with open('win22-requests-anon.csv', newline='') as csvfile:
  student_courses_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
  row_number = -1
  for row in student_courses_reader:
    row_number += 1
    if (row[0] == "Name" or len(row[1]) < 2): # skip header row or if row empty
      continue

    student = row[0]
    student_to_courses_dict[student] = []
    student_to_csv_rows_dict[student] = row_number
    for i in range(1,6):
      course = row[i]
      if (len(course) < 2): # skip when blank row (no class chosen)
        continue

      x = model.addVar(vtype=GRB.BINARY, name=student + "_" + course)
      major = row[7]
      degree = row[8]

      student_to_major_dict[student] = major
      student_to_degree_dict[student] = degree
      priority_dict_key = major + " " + degree
      student_priority = priorities[priority_dict_key] if (priority_dict_key in priorities) else 1
      obj.addTerms(student_priority * (6-i), x)

      student_to_courses_dict[student].append(course)
      
      if (row[8][:2] == "MS"):
        if course in course_to_MS_students_dict:
          course_to_MS_students_dict[course].append(student)
        else:
          course_to_MS_students_dict[course] = [student]

      if course in course_to_students_dict:
          course_to_students_dict[course].append(student)
          if (i in course_stats[course]):
            course_stats[course][i] += 1
          else:
            course_stats[course][i] = 1
      else:
        course_to_students_dict[course] = [student]
        course_stats[course] = {}
        course_stats[course][i] = 1
        
    dummy_var = model.addVar(vtype=GRB.BINARY, name=student + "_dummy") # dummy variable to facilitate fairness
    obj.addTerms(0.5, dummy_var)
    dummy_vars.append(dummy_var)

    model.update()
    this_students_courses = student_to_courses_dict[student]
    this_student_vars = [model.getVarByName(student + "_" + course) for course in this_students_courses]
    student_to_vars_dict[student] = this_student_vars
    model.addConstr(sum(this_student_vars) + dummy_var <= credit_cap + 0.5) # student credit cap

    # want CS students to get at least 2 of their top 4 classes
    if (major == "CS"): 
      model.addConstr(sum(this_student_vars[:4]) + dummy_var >= min(len(this_student_vars), 2))
       
    for course in this_students_courses:
      conflicts = set(filter(lambda x: len(parse_course_time(x).intersection(parse_course_time(course))) != 0, this_students_courses))
      conflict_vars = [model.getVarByName(student + "_" + course) for course in conflicts] if len(conflicts) > 1 else []
      model.addConstr(sum(conflict_vars) <= 1) # time conflict constraint

  for course in course_to_students_dict:
    this_course_potential_students = course_to_students_dict[course]
    this_course_potential_MS_students = course_to_MS_students_dict[course]
    course_vars = [model.getVarByName(student + "_" + course) for student in this_course_potential_students]
    course_MS_vars = [model.getVarByName(student + "_" + course) for student in this_course_potential_MS_students]
    model.addConstr(sum(course_vars) <= course_capacities[course]) # course enrollment capacity constraint
  
  model.addConstr(sum(dummy_vars) <= 0.185 * len(dummy_vars)) # controlling how lax the fairness constraint is
    
model.setObjectiveN(obj, 0, 1)
model.ModelSense = GRB.MAXIMIZE
model.optimize()

with open('course_enrollments.csv', 'w', newline='') as enrollment:
  enrollment_writer = csv.writer(enrollment, delimiter=',', quotechar='|')
  enrollment_writer.writerow(["course", "# of students enrolled", "course capacity"])
  for course in course_to_students_dict:
    student_assignment = [model.getVarByName(student + "_" + course).x for student in course_to_students_dict[course]]
    enrollment_writer.writerow([course, sum(student_assignment), course_capacities[course]])

with open('win22-requests-anon.csv', newline='') as csvfile, open('assignments.csv', 'w', newline='') as assignments_file:
  student_courses_reader = csv.reader(csvfile, delimiter=',', quotechar='|')
  assignment_writer = csv.writer(assignments_file, delimiter=',', quotechar='|')
  row_number = -1
  for row in student_courses_reader:
    row_number += 1
    if (row_number == 0):
      assignment_writer.writerow(row + ["# of courses gotten", "got first choice?", "got second choice?", "got third choice?", "got fourth choice?", "got fifth choice?"])
    else:
      student = row[0]
      if (student not in student_to_courses_dict): continue
      courses = student_to_courses_dict[student]
      assignments = [int(model.getVarByName(student + "_" + course).x) for course in courses]
      
      if (row_number == student_to_csv_rows_dict[student]):
        assignment_writer.writerow(row + [sum(assignments)] + assignments)
      else:
        assignment_writer.writerow(row)