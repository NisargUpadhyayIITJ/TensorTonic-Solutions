-- Write your SQL query here
SELECT name, salary, dept_name
FROM employees INNER JOIN departments ON employees.dept_id = departments.id
ORDER BY name ASC;