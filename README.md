# rna_ai_builder
เนื่องจาก ไฟล์ มัน ใหญ่ ผมจึง ลอง ทำกับ ไฟล์ เดียวก่อน ผม run ใน WSL ครับ เพราะ ส่วน ใหญ่  tools ที่ผมใช้ ทำงาน ใน Linux
Pipeline
1. QC ก่อน เพื่อรู้ว่าข้อมูลมีปัญหาอะไร ก่อนตัดสินใจ trim ซึ่งไม่มีปัญหา
2. Trim ตัด adapter + low-quality bases ออก reads จาก 38.7M → 26.6M
3. qc อีกรอบ ซึ่งไม่มีปัญหา

ต่อไป หลังจาก check ข้อมูล เรา จะ STAR(Spliced Transcripts Alignment to a Reference) กัน
1. Download genome + annotation เพื่อ เอา reads จาก FASTQ ของเราไปเทียบ ว่า แต่ละตัวมาจากตำแหน่งไหนในจีโนม (ลายละเอียดอยู่ใน foder Download genome + annotation)
2. build genome index มันคือการ จัดระเบียบ genome ใหม่เพื่อ ให้ค้นหาได้เร็ว
3. จากนั้น เราก็ Align ได้เลย
4. แล้วก็ run จับ exon count กับ sample ได้เลย

แล้วจากนั้นก็ แปลง ไฟล์เข้าไปอยู่ในตราง

ในขั้น ตอนนี้ ถือ ว่า data eda เสร็จแล้ว แต่ เรา ยัง ต้อง เลือก Feature ก่อน เพราะ มี 2,164,410  Feature(exons) ซึ่ง exon ส่วนใหญ่ จะไม่มีประโยชน์ ส่วนวิธีที่ ผมจะใช้คือ
1.เอา exon ที่ไม่มี การเปลี่ยน แปลง ออก
2.ใน ขั้น ตอนนี้ ผมจะ Normalize  เลย ด้วยวิธีการ  log1p เพื่อไม่ให้ ค่าต่างกัน เกิน
3.เอา exon ที่ เปลี่ยน แปลง 5000 อันดับแรกออกมา
1. Download genome + annotation เพื่อ เอา reads จาก FASTQ ของเราไปเทียบ ว่า แต่ละตัวมาจากตำแหน่งไหนในจีโนม (ลายละเอียดอยู่ใน foder Download genome + annotation)
2. build genome index มันคือการ จัดระเบียบ genome ใหม่เพื่อ ให้ค้นหาได้เร็ว
3. จากนั้น เราก็ Align ได้เลย
4. แล้วก็ run จับ exon count กับ sample ได้เลย
