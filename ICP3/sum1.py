class Employee:
    number_of_employees = 0           # creating a class employee and count for no of employees
    total_salary = 0

    def __init__(self, name, age, salary, department):
        self.name = name                                           #Constructor to intialize values
        self.age = age
        self.salary = salary
        self.department = department

        Employee.number_of_employees += 1
        Employee.total_salary += self.salary

    def getAverageSalary():                                             #Function to calculate the average salaries
        avg = Employee.total_salary / Employee.number_of_employees
        return round(avg, 2)


class FulltimeEmployee(Employee):               #class FulltimeEmployee and inheriting the properties of employee class

    def __init__(self, name, age, salary, department):
        Employee.__init__(self, name, age, salary, department)

    def createObjects():
        print("--------Employee Database---------\n")

        while True:
            bool = input("\n Do you want to enter employee details (yes/no): ")

            if bool == "yes":
                name = input("Enter Employee name: ")
                age = int(input("Enter Employee age: "))
                salary = int(input("Enter Employee salary: "))
                department = input("Enter Employee Department: ")

                FulltimeEmployee(name, age, salary, department)

            else:
                print("Total Number of Employees: " + str(Employee.number_of_employees))
                print("Final Average Salary of Employees: " + str(Employee.getAverageSalary()))
                break


if __name__ == "__main__":
    FulltimeEmployee.createObjects()