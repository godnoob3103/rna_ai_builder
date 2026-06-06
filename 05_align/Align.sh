STAR --runThreadN 16 \
     --genomeDir ~/reference/star_index \
     --readFilesIn /mnt/a/data\ sci/data/trimmed/cancer/ERR164550_1.fastq.gz \
                   /mnt/a/data\ sci/data/trimmed/cancer/ERR164550_2.fastq.gz \
     --readFilesCommand zcat \
     --outSAMtype BAM SortedByCoordinate \
     --outFileNamePrefix ~/aligned/cancer/ERR164550_ \
     --outSAMattributes NH HI AS NM
