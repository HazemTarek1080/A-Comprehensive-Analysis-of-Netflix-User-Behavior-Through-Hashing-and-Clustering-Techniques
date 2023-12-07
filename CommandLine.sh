awk -F, '$3 != 0 {print $4}' vodclickstream_uk_movies_03.csv | sort | uniq -c| sort -nr| head -n 1
seconds=$(awk -F, '{sum+=$3} END {print sum/NR}' vodclickstream_uk_movies_03.csv)
echo "The average time between subsequent clicks on Netflix.com in seconds is: $seconds"
user=$(awk -F, '{sum[$NF]+=$3} END {max=0; for (i in sum) if (sum[i]>max) {max=sum[i]; max_id=i} print max_id}' vodclickstream_uk_movies_03.csv)
echo "The ID of the user that has spent the most time on Netflix is: $user"
