# rna_ai_builder
เนื่องจาก ไฟล์ มัน ใหญ่ ผมจึง ลอง ทำกับ ไฟล์ เดียวก่อน ผม run ใน WSL ครับ เพราะ ส่วน ใหญ่  tools ที่ผมใช้ ทำงาน ใน Linux
Pipeline
1. QC ก่อน เพื่อรู้ว่าข้อมูลมีปัญหาอะไร ก่อนตัดสินใจ trim ซึ่งไม่มีปัญหา
2. Trim ตัด adapter + low-quality bases ออก reads จาก 38.7M → 26.6M
3. qc อีกรอบ ซึ่งไม่มีปัญหา 

ต่อไป หลังจาก check ข้อมูล 
Download genome + annotation เพื่อ เอา reads จาก FASTQ ของเราไปเทียบ ว่า แต่ละตัวมาจากตำแหน่งไหนในจีโนม (ลายละเอียดอยู่ใน foder Download genome + annotation)
