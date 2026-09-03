SELECT customerId, customerName,
       vendorSuggestCustomers, vendorSuggestCustomersName,
       virtualCustomers, virtualCustomersName,
       day, nodeId, state
FROM node_day_ops_wide_full
WHERE customerId IN (10000251,10000182,10000193,10000184)
ORDER BY day DESC
LIMIT 20
