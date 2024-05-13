# It's a Fraud
Given details about a transaction determine whether the transaction is a Fraud or Not.
# Columns

* TransactionDT: timedelta from a given reference datetime (not an actual timestamp) “TransactionDT "corresponds to the number of seconds in a day.

* TransactionAMT: transaction payment amount in USD

* ProductCD: product code, the product for each transaction

* card1 - card6: payment card information, such as card type, card category, issue bank, country, etc.

* addr: address 
  “both addresses are for purchaser
   addr1 as billing region
   addr2 as billing country”
* dist: distances between (not limited) billing address, mailing address, zip code, IP address, phone area, etc.”

* P_ and (R__) emaildomain: purchaser and recipient email domain
    “ certain transactions don't need recipient, so R_emaildomain is null.”
    
* C1-C14: counting, such as how many addresses are found to be associated with the payment card, etc.

* D1-D15: timedelta, such as days between previous transaction, etc.

* M1-M9: match, such as names on card and address, etc.

* Vxxx: Vesta engineered rich features, including ranking, counting, and other entity relations.

* id01 to id11 are numerical features for identity.

* Other columns such as network connection information (IP, ISP, Proxy, etc),digital signature (UA/browser/os/version, etc) associated with transactions are also present.

# Instructions to run the Code for any py file

* Make sure you have all the required packages installed (Check the config File incase)

* To install a package, general command is as follows - pip3 install packagename

* To run the code, run command - python3 pythonfilename
