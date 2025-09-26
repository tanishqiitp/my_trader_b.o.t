
SELECT version();
\c mydatabase;

CREATE TABLE employees (
    id SERIAL PRIMARY KEY,
    name VARCHAR(50),
    position VARCHAR(50),
    salary NUMERIC(10,2),
    hire_date DATE DEFAULT CURRENT_DATE
);
INSERT INTO employees (name, position, salary)
VALUES
    ('Alice', 'Manager', 75000),
    ('Bob', 'Developer', 60000),
    ('Charlie', 'Analyst', 50000);

