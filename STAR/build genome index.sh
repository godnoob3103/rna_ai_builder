STAR --runThreadN 16 \
     --runMode genomeGenerate \
     --genomeDir ~/reference/star_index \
     --genomeFastaFiles ~/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa \
     --sjdbGTFfile ~/reference/Homo_sapiens.GRCh38.113.gtf \
     --sjdbOverhang 100
