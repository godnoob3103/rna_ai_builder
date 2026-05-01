#!/bin/bash

DATA="/mnt/a/data sci/data"
REF=~/reference/star_index
GTF=~/reference/Homo_sapiens.GRCh38.113.gtf
ALIGNED=~/aligned
TRIMMED="$DATA/trimmed"
COUNTS="$DATA/counts"

mkdir -p "$TRIMMED/cancer" "$TRIMMED/non_cancer"
mkdir -p "$ALIGNED/cancer" "$ALIGNED/non_cancer"
mkdir -p "$COUNTS"

# ฟังก์ชัน process แต่ละ sample
process_sample() {
    local sample=$1
    local label=$2  # cancer หรือ non_cancer

    echo "===== Processing $sample ($label) ====="

    # Trim
    fastp \
        -i "$DATA/$label/${sample}_1.fastq.gz" \
        -I "$DATA/$label/${sample}_2.fastq.gz" \
        -o "$TRIMMED/$label/${sample}_1.fastq.gz" \
        -O "$TRIMMED/$label/${sample}_2.fastq.gz" \
        -h "$TRIMMED/$label/${sample}_report.html" \
        -j "$TRIMMED/$label/${sample}_report.json" \
        --thread 16 2>&1

    # Align
    rm -rf /tmp/STAR_${sample}
    STAR --runThreadN 16 \
        --genomeDir $REF \
        --readFilesIn "$TRIMMED/$label/${sample}_1.fastq.gz" \
                      "$TRIMMED/$label/${sample}_2.fastq.gz" \
        --readFilesCommand zcat \
        --outSAMtype BAM SortedByCoordinate \
        --outFileNamePrefix "$ALIGNED/$label/${sample}_" \
        --outSAMattributes NH HI AS NM \
        --outTmpDir /tmp/STAR_${sample} 2>&1

    # Count exons
    featureCounts \
        -T 16 -p -P -B -f -O \
        -a $GTF \
        -o "$COUNTS/${sample}_exon_counts.txt" \
        "$ALIGNED/$label/${sample}_Aligned.sortedByCoord.out.bam" 2>&1

    echo "===== Done $sample ====="
}

# Cancer samples (ข้าม ERR164550 เพราะทำแล้ว)
for sample in ERR164557 ERR164562 ERR164564 ERR164565 ERR164569 ERR164570 ERR164572 ERR164574 ERR164575 ERR164578; do
    process_sample $sample cancer
done

# Non-cancer samples
for sample in ERR164476 ERR164479 ERR164480 ERR164484 ERR164487 ERR164491 ERR164497 ERR164498 ERR164502 ERR164503 ERR164504 ERR164507; do
    process_sample $sample non_cancer
done

echo "===== ALL DONE ====="
EOF

chmod +x ~/run_pipeline.sh
