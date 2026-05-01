STAR --runThreadN 16 \
     --runMode genomeGenerate \
     --genomeDir ~/reference/star_index \
     --genomeFastaFiles ~/reference/Homo_sapiens.GRCh38.dna.primary_assembly.fa.gz \
     --sjdbGTFfile ~/reference/Homo_sapiens.GRCh38.113.gtf.gz \
     --sjdbOverhang 100
