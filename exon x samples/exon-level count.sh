featureCounts \
  -T 16 \
  -p -P -B \
  -f -O \
  -a ~/reference/Homo_sapiens.GRCh38.113.gtf \
  -o ~/aligned/cancer/ERR164550_exon_counts.txt \
  ~/aligned/cancer/ERR164550_Aligned.sortedByCoord.out.bam
