class Parent:
    def __init__(self, name ):
        self.name = name

    def say_hello(self):
        print(f'Hello, {self.name}!')

    @classmethod
    def construct_from_name_path(cls, name, age):
        print(f'cls {cls}')
        return cls(name, age)

class Child(Parent):
    def __init__(self, name, age):
        super().__init__(name)
        self.age = age

    def say_hello(self):
        print(f'Hello, {self.name}! I am {self.age} years old.')

if __name__ == '__main__':
    child = Child.construct_from_name_path("John", 100)
    child.say_hello()