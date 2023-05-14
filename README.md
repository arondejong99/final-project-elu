# final-project-elu
This is the final project of my master in data science for the European Leadership University (ELU). My project is about creating an algorithm that could detect lung cancer based on CT scans.

Ultimately, the following predictions will be done:
- Predict if the patient has cancer
- What type of cancer the patient has if applied
- The size of the tumor

For first prediction, you would need the following zip file: normal_ct_scans.zip
You would also need a subselection of the data in the pcia file: Lung-PET-CT-Dx-NBIA-Manifest-122220.tcia

For getting a subselection of the tcia file, the following patientIds are used:
['Lung_Dx-A0002' 'Lung_Dx-A0003' 'Lung_Dx-A0004' 'Lung_Dx-A0005'
 'Lung_Dx-A0007' 'Lung_Dx-A0009' 'Lung_Dx-A0010' 'Lung_Dx-A0016'
 'Lung_Dx-A0021' 'Lung_Dx-A0024' 'Lung_Dx-A0025' 'Lung_Dx-A0026'
 'Lung_Dx-A0027' 'Lung_Dx-A0034' 'Lung_Dx-A0038' 'Lung_Dx-A0043'
 'Lung_Dx-A0044' 'Lung_Dx-A0045' 'Lung_Dx-A0047' 'Lung_Dx-A0048'
 'Lung_Dx-A0050' 'Lung_Dx-A0051' 'Lung_Dx-A0052' 'Lung_Dx-A0056'
 'Lung_Dx-A0062' 'Lung_Dx-A0068' 'Lung_Dx-A0073' 'Lung_Dx-A0074'
 'Lung_Dx-A0081' 'Lung_Dx-B0002' 'Lung_Dx-B0004' 'Lung_Dx-B0005'
 'Lung_Dx-B0006' 'Lung_Dx-B0011' 'Lung_Dx-B0012' 'Lung_Dx-B0013'
 'Lung_Dx-B0014' 'Lung_Dx-B0015' 'Lung_Dx-B0016' 'Lung_Dx-B0017'
 'Lung_Dx-B0018' 'Lung_Dx-B0019' 'Lung_Dx-B0020' 'Lung_Dx-B0022'
 'Lung_Dx-B0023' 'Lung_Dx-B0024' 'Lung_Dx-B0026' 'Lung_Dx-B0027'
 'Lung_Dx-B0028' 'Lung_Dx-B0031' 'Lung_Dx-B0044' 'Lung_Dx-E0001'
 'Lung_Dx-E0002' 'Lung_Dx-E0003' 'Lung_Dx-E0004' 'Lung_Dx-E0005'
 'Lung_Dx-G0001' 'Lung_Dx-G0002' 'Lung_Dx-G0003' 'Lung_Dx-G0004'
 'Lung_Dx-G0005' 'Lung_Dx-G0006' 'Lung_Dx-G0007' 'Lung_Dx-G0008'
 'Lung_Dx-G0009' 'Lung_Dx-G0010' 'Lung_Dx-G0011' 'Lung_Dx-G0012'
 'Lung_Dx-G0013' 'Lung_Dx-G0014' 'Lung_Dx-G0015' 'Lung_Dx-G0016']

 ## Get started
When you clone the repository into your local machine, ou need to make sure the CDM files are stored there. It couldn't be stored in the github repo because of size issues. It was also impossible to zip the files and store them from there.

To succesfully store the CDM files, take the following steps.
1. Download the NBIA Data Retriever. The installation guide can be found here: https://wiki.cancerimagingarchive.net/display/NBIA/Downloading+TCIA+Images
2. Open the tcia file using the NBIA Data Retriever.
3. Download the files from the tcia file into your local machine. Be sure to store them in the following folder: `tciaDownload\manifest-1608669183333\Lung-PET-CT-Dx`. In this folder, the files are stored using the following folder format: [patient_id]/[date-NA-body-type]/[ct-scan-details]
4. Once the files are stored, remove the folders with a patientId that doesn't match from the list mentioned above.