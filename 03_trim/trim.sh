fastp \
  -i cancer/ERR164550_1.fastq.gz \
  -I cancer/ERR164550_2.fastq.gz \
  -o trimmed/cancer/ERR164550_1.fastq.gz \
  -O trimmed/cancer/ERR164550_2.fastq.gz \
  -h trimmed/cancer/ERR164550_report.html \
  -j trimmed/cancer/ERR164550_report.json \
  --thread 4
