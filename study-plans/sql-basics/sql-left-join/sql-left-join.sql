-- Write your SQL query here
SELECT name, city, COALESCE(SUM(amount), 0) as total_spent
FROM customers LEFT JOIN orders ON customers.id = orders.customer_id
GROUP BY name, city
ORDER BY total_spent DESC, name ASC;
