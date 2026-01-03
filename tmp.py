<!doctype html>
<meta charset="utf-8">
<title>水下机器人</title>
<link href="style.css" rel="stylesheet" type="text/css" media="all" />
<script src="jquery-3.2.0.min.js" charset="utf-8" type="text/javascript"></script>
<script src="echarts.min.js" charset="utf-8" type="text/javascript"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>

<style>
    /* 增强功能开关样式 */
    .enhancement-toggle {
        position: absolute;
        top: 10px;
        right: 10px;
        z-index: 1000;
        background: rgba(0, 0, 0, 0.7);
        padding: 10px 15px;
        border-radius: 5px;
        display: flex;
        align-items: center;
        gap: 10px;
    }
    
    .enhancement-toggle label {
        color: #fff;
        font-size: 14px;
        margin: 0;
        cursor: pointer;
        user-select: none;
    }
    
    .toggle-switch {
        position: relative;
        display: inline-block;
        width: 50px;
        height: 24px;
    }
    
    .toggle-switch input {
        opacity: 0;
        width: 0;
        height: 0;
    }
    
    .slider {
        position: absolute;
        cursor: pointer;
        top: 0;
        left: 0;
        right: 0;
        bottom: 0;
        background-color: #ccc;
        transition: .4s;
        border-radius: 24px;
    }
    
    .slider:before {
        position: absolute;
        content: "";
        height: 18px;
        width: 18px;
        left: 3px;
        bottom: 3px;
        background-color: white;
        transition: .4s;
        border-radius: 50%;
    }
    
    input:checked + .slider {
        background-color: #4caf50;
    }
    
    input:checked + .slider:before {
        transform: translateX(26px);
    }
    
    .enhancement-status {
        position: absolute;
        top: 45px;
        right: 10px;
        z-index: 999;
        background: rgba(76, 175, 80, 0.8);
        color: white;
        padding: 5px 10px;
        border-radius: 3px;
        font-size: 12px;
        display: none;
    }
    
    .enhancement-status.active {
        display: block;
    }
</style>

<body>
    <div class="wpbox">
        <div class="bnt">
            <div class="topbnt_left fl">
                <ul>
                    <li class="active"><a href="javascript:void(0)" onclick="window.location.reload();">实例分割</a></li>
                    <li><a href="#">目标检测</a></li>
                    <li><a href="#">语义分割</a></li>
                </ul>
            </div>
            <h1 class="tith1 fl">水下机器人</h1>
            <div class=" fr topbnt_right">
                <ul>
                    <li><a href="#">to do</a></li>
                    <li><a href="#">to do</a></li>
                    <li class="active"><a href="#">数据分析</a></li>
                </ul>
            </div>
        </div>
        <!-- bnt end -->
        <div class="left1" style="width:21%">
            <div class="aleftboxttop" >
                <h2 class="tith2">设备电量</h2>
                <div id="chart-container" style="width: 80%; margin: 10% auto;height: 50%;display: flex;justify-content: center;align-items: center;padding: 0px;">
                    <canvas id="batteryChart" style="height:100%"></canvas>
                </div>
            </div>
            <div class="aleftboxtmidd">
                <h2 class="tith2" style="font-family: 'Arial'">实例统计</h2>
                <div class="lefttoday_tit height ht">
                    <p class="fl">地区：连心渔港</p>
                    <p class="fr" id="timePeriod2">时间段：2024-12月</p>
                </div>
                <div id="aleftboxtmidd" class="aleftboxtmiddcont">
                    <canvas id="categoryChart" style="height: 100%;width:100%"></canvas>
                </div>
            </div>
            <div class="aleftboxtbott" style="padding: 0%;width: 97%;" onclick="runInference()">
                <h2 class="tith2">截图</h2>
                <img id="snapshot" alt="Captured Snapshot" src="./image_0127.jpg" style="width:90%;height:90%;"/>
            </div>
        </div>
        <!--  left1 end -->
        <div class="mrbox" style="width:75%;">
            <div class="mrbox_topmidd" style="width: 68%;margin-top: 3%;">
                <div class="amiddboxttop" style="text-align: center;height: 80%;position: relative;">
                    <h2 class="tith2 pt1" style="text-align: center;margin-top: 1.5%;">视频流</h2>
                    
                    <!-- 图像增强开关 -->
                    <div class="enhancement-toggle">
                        <label for="enhancementToggle">图像增强</label>
                        <label class="toggle-switch">
                            <input type="checkbox" id="enhancementToggle">
                            <span class="slider"></span>
                        </label>
                    </div>
                    
                    <!-- 增强状态提示 -->
                    <div class="enhancement-status" id="enhancementStatus">
                        增强中...
                    </div>
                    
                    <div class="amiddboxttop_map" onclick="captureSnapshot()" style="margin:0.6% 5% 0 5%; width: 90.3%;">
                        <img id="streamimage1" class="xform" src="./image_0127.jpg" alt="Streaming video 1" style="width: 100%; height: 100%; object-fit: cover; "/>
                    </div>
                </div>
                <!--  amiddboxttop end-->
            </div>
            <!-- mrbox_top end -->
            <div class="mrbox_top_right" style="width:30%">
                <div class="arightboxtop">
                    <h2 class="tith2">结果分析</h2>
                    <div class="lefttoday_tit">
                        <p class="fl">地区：连心渔港</p>
                        <p class="fr" id="timePeriod">时间段：2024-12-20</p>
                    </div>
                    <div class="zhtc_table_title">
                        <div>类别</div>
                        <div>置信度</div>
                    </div>
                    <div id="zhtc_table" class="left2_table">
                        <ul id="segmentation-results" style="max-height: 88%; overflow-y: auto;padding: 10px;list-style-type: none;margin: 0;">
                            <li>
                                <div class="zhtc_table_li_content">
                                    <div>ship_hull</div>
                                    <div>99.5%</div>
                                </div>
                            </li>
                        </ul>
                    </div>
                </div>
                <div class="arightboxbott" style="text-align: center;">
                    <h2 class="tith2 ">分割结果</h2>
                    <img id="imgseg_result" alt="Inference Result" src="data:image/gif;base64,R0lGODlhAQABAAD/ACwAAAAAAQABAAACADs=" style="width:90%;height: 88%;"/>
                </div>
            </div>
            <!-- mrbox_top_right end -->
        </div>
    </div>

    <script type="text/javascript">
        // 全局变量
        let snapshotBlob = null;
        let chartInstance = null;
        let isEnhancementEnabled = false;
        let currentImagePath = './image_0127.jpg';
        
        const categories = [
            'sea_chest_grating', 'paint_peel', 'over_board_valve', 'defect',
            'corrosion', 'propeller', 'anode', 'bilge_keel', 'marine_growth', 'ship_hull'
        ];
        
        // 监听增强开关变化
        document.getElementById('enhancementToggle').addEventListener('change', function(e) {
            isEnhancementEnabled = e.target.checked;
            console.log('Image enhancement:', isEnhancementEnabled ? 'enabled' : 'disabled');
            
            // 如果开启增强，重新处理当前图像
            if (isEnhancementEnabled) {
                processWithEnhancement(currentImagePath);
            } else {
                // 如果关闭增强，恢复原始图像并重新推理
                document.getElementById('streamimage1').src = currentImagePath;
                runInference(currentImagePath, false);
            }
        });
        
        window.onload = function() {
            captureSnapshot(function() {
                runInference(currentImagePath, false);
            });
        };
        
        function captureSnapshot(callback) {
            const xhr = new XMLHttpRequest();
            xhr.open('GET', './?action=snapshot', true);
            xhr.setRequestHeader('Authorization', 'Basic ' + btoa('your_username:your_password'));

            xhr.onload = function () {
                if (xhr.status === 200) {
                    snapshotBlob = xhr.response;
                    const imgUrl = URL.createObjectURL(xhr.response);
                    document.getElementById('snapshot').src = './image_0127.jpg';
                    callback();
                } else {
                    console.error('Error fetching snapshot:', xhr.status);
                }
            };

            xhr.onerror = function () {
                console.error('Network error');
            };

            xhr.responseType = 'blob';
            xhr.send();
        }
        
        // 图像增强处理函数
        function processWithEnhancement(imagePath) {
            // 显示增强状态
            const statusDiv = document.getElementById('enhancementStatus');
            statusDiv.classList.add('active');
            
            fetch(imagePath)
                .then(response => response.blob())
                .then(blob => {
                    const formData = new FormData();
                    formData.append('image', blob, 'snapshot.jpg');
                    formData.append('resize', '0');
                    formData.append('return_maps', 'false');
                    
                    // 调用增强API
                    return fetch('http://10.142.15.54:5002/enhance', {
                        method: 'POST',
                        body: formData
                    });
                })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        // 显示增强后的图像
                        const enhancedImg = `data:image/png;base64,${data.enhanced_image}`;
                        document.getElementById('streamimage1').src = enhancedImg;
                        
                        // 使用增强后的图像进行分割
                        runInference(imagePath, true, data.enhanced_image);
                    } else {
                        console.error('Enhancement failed:', data.error);
                        alert('图像增强失败: ' + data.error);
                    }
                    
                    // 隐藏增强状态
                    statusDiv.classList.remove('active');
                })
                .catch(error => {
                    console.error('Error during enhancement:', error);
                    alert('图像增强出错: ' + error.message);
                    statusDiv.classList.remove('active');
                });
        }
        
        // 修改后的推理函数
        function runInference(imagePath, useEnhanced, enhancedBase64) {
            imagePath = imagePath || './image_0127.jpg';
            
            // 如果启用增强但还没有增强图像，先进行增强
            if (isEnhancementEnabled && !useEnhanced) {
                processWithEnhancement(imagePath);
                return;
            }
            
            // 准备要发送的图像数据
            let imagePromise;
            
            if (useEnhanced && enhancedBase64) {
                // 使用增强后的base64图像
                imagePromise = fetch(`data:image/png;base64,${enhancedBase64}`)
                    .then(response => response.blob());
            } else {
                // 使用原始图像
                imagePromise = fetch(imagePath)
                    .then(response => response.blob());
            }
            
            imagePromise
                .then(blob => {
                    const formData = new FormData();
                    formData.append('image', blob, 'snapshot.jpg');
                    
                    const xhr = new XMLHttpRequest();
                    xhr.open('POST', 'http://10.142.15.54:5001/process-snapshot', true);
                    xhr.responseType = 'json';
                    
                    xhr.onload = function () {
                        if (xhr.status === 200) {
                            console.log(xhr.response);
                            const data = xhr.response;
                            const img = new Image();
                            img.src = `data:image/jpeg;base64,${data.image_data}`;
                            document.getElementById('imgseg_result').src = img.src;
                            
                            // 更新分割结果表格
                            const ul = document.getElementById("segmentation-results");
                            const segResult = data.seg_result;
                            ul.innerHTML = '';
                            segResult.labels.forEach((label, index) => {
                                const category = categories[label];
                                const li = document.createElement("li");
                                li.innerHTML = `
                                    <div class="zhtc_table_li_content">
                                        <div>${category}</div>
                                        <div>${(segResult.scores[index] * 100).toFixed(1) + '%'}</div>
                                    </div>
                                `;
                                ul.appendChild(li);
                            });
                            
                            // 统计各类别的总数
                            const categoryCounts = {};
                            segResult.labels.forEach(label => {
                                const category = categories[label];
                                categoryCounts[category] = (categoryCounts[category] || 0) + 1;
                            });
                            
                            // 准备柱状图数据
                            const labels = Object.keys(categoryCounts);
                            const chart_data = Object.values(categoryCounts);
                            
                            // 创建柱状图
                            const ctx = document.getElementById('categoryChart').getContext('2d');
                            if (chartInstance) {
                                chartInstance.destroy();
                            }
                            chartInstance = new Chart(ctx, {
                                type: 'bar',
                                data: {
                                    labels: labels,
                                    datasets: [{
                                        label: '类别计数',
                                        data: chart_data,
                                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                                        borderColor: 'rgba(75, 192, 192, 1)',
                                        borderWidth: 1
                                    }]
                                },
                                options: {
                                    responsive: true,
                                    scales: {
                                        y: {
                                            beginAtZero: true,
                                            ticks: {
                                                color: '#fff',
                                                font: {
                                                    size: 12,
                                                    weight: 'bold'
                                                }
                                            }
                                        },
                                        x: {
                                            ticks: {
                                                color: '#fff',
                                                font: {
                                                    size: 10,
                                                    weight: 'normal'
                                                }
                                            }
                                        }
                                    },
                                    plugins: {
                                        legend: {
                                            position: 'top',
                                            labels: {
                                                color: '#fff',
                                                font: {
                                                    size: 12,
                                                    weight: 'bold'
                                                }
                                            },
                                            onClick: function(event, legendItem) {
                                                event.stopPropagation();
                                            }
                                        }
                                    }
                                }
                            });
                        } else {
                            console.error('Error during inference:', xhr.status, xhr.statusText);
                        }
                    };
                    
                    xhr.onerror = function () {
                        console.error('Network error');
                    };
                    
                    xhr.send(formData);
                })
                .catch(error => {
                    console.error("Error fetching image:", error);
                });
        }
        
        // 电池电量显示
        let batteryData = {
            remaining: 100,
            used: 0
        };
        
        const ctx_battery = document.getElementById('batteryChart').getContext('2d');
        const batteryChart = new Chart(ctx_battery, {
            type: 'pie',
            data: {
                labels: ['剩余电量', '已用电量'],
                datasets: [{
                    data: [batteryData.remaining, batteryData.used],
                    backgroundColor: ['#4caf50', '#ff6347']
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'right',
                        align: 'center',
                        labels: {
                            font: {
                                size: 14
                            }
                        },
                        onClick: function(event, legendItem) {
                            event.stopPropagation();
                        }
                    }
                }
            }
        });
        
        function decreaseBattery() {
            if (batteryData.remaining > 0) {
                batteryData.remaining -= 1;
                batteryData.used = 100 - batteryData.remaining;
                batteryChart.data.datasets[0].data = [batteryData.remaining, batteryData.used];
                batteryChart.update();
            } else {
                batteryData = {
                    remaining: 100,
                    used: 0
                };
            }
        }
        
        const batteryInterval = setInterval(decreaseBattery, 1000);
        
        // 日期显示
        const currentDate = new Date();
        const year = currentDate.getFullYear();
        const month = (currentDate.getMonth() + 1).toString().padStart(2, '0');
        const day = currentDate.getDate().toString().padStart(2, '0');
        const formattedDate = `${year}-${month}-${day}`;
        
        document.getElementById('timePeriod').textContent = `时间段：${formattedDate}`;
        document.getElementById('timePeriod2').textContent = `时间段：${year}-${month}月`;
    </script>
</body>
</html>
