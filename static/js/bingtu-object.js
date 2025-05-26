var dom = document.getElementById('container-object');
var myChart = echarts.init(dom, null, {
  renderer: 'svg',
  useDirtyRect: false
});
var app = {};

var option;

var data = [
{
    name: 'OCR in \n The Wild',
    // value: 1.1,
    itemStyle: {
      color: '#b14d57'
    },
    children: [
      {
        name: 'Contact Information',
        value: 1,
        itemStyle: {
          color: '#c78936'
        }
      },
      {
        name: 'Product & Advertisement',
        value: 1,
        itemStyle: {
          color: '#8c292c'
        }
      },
      {
        name: 'Identity Information',
        value: 1,
        itemStyle: {
          color: '#e5762e'
        }
      },
      {
        name: 'Book, Map & Poster',
        value: 1,
        itemStyle: {
          color: '#dd4c51'
        }
      },
      {
        name: 'Signage & Other Text',
        value: 1,
        itemStyle: {
          color: '#a16c5a'
        }
      },
      {
        name: 'Scene Recognition',
        value: 1,
        itemStyle: {
          color: '#E5CCCC'
        }
      },
      {
        name: 'Characters Understanding',
        value: 1,
        itemStyle: {
          color: '#E0C0C0'
        }
      }
    ]
  },
  {
    name: 'Remote Sensing',
    // value: 1.1,
    itemStyle: {
      color: '#dd4c51'
    },
    children: [
      {
        name: 'Object Counting',
        value: 1,
        itemStyle: {
          color: '#3e0317'
        }
      },
      {
        name: 'Spatial Relationship',
        value: 1,
        itemStyle: {
          color: '#e62969'
        }
      },
      {
        name: 'Color Recognition',
        value: 1,
        itemStyle: {
          color: '#ef2d36'
        }
      },
      ]
  },
  {
    name: 'Diagram \nTable',
    // value: 1.1,
    itemStyle: {
      color: '#e65832'
    },
            children: [
      {
        name: 'Diagram Perception',
        value: 1,
        itemStyle: {
          color: '#d45a59'
        }
      },
      {
        name: 'Table Perception',
        value: 1,
        itemStyle: {
          color: '#ae341f'
        }
      },
      {
        name: 'Calculation (Table)',
        value: 1,
        itemStyle: {
          color: '#E5E5CC'
        }
      },
      {
        name: 'Calculation (Diagram)',
        value: 1,
        itemStyle: {
          color: '#E0E0C0'
        }
      },
      {
        name: 'Comparision (Diagram)',
        value: 1,
        itemStyle: {
          color: '#CCCC99'
        }
      },
      {
        name: 'Comparision (Table)',
        value: 1,
        itemStyle: {
          color: '#C0C080'
        }
      },
      {
        name: 'Forecasting (Diagram)',
        value: 1,
        itemStyle: {
          color: '#B2B266'
        }
      },
      {
        name: 'Forecasting (Table)',
        value: 1,
        itemStyle: {
          color: '#E5E5CC'
        }
      },
      ]
  },
  {
    name: 'Autonomous \nDriving',
    // value: 1.1,
    itemStyle: {
      color: '#a87b64'
    },
            children: [
      {
        name: 'Identity Perception',
        value: 1,
        itemStyle: {
          color: '#c78869'
        }
      },
      {
        name: 'Vehicle Motion',
        value: 1,
        itemStyle: {
          color: '#d4ad12'
        }
      },
      {
        name: 'Multi-Vehicle Motion',
        value: 1,
        itemStyle: {
          color: '#9d5433'
        }
      },
      {
        name: 'Pedestrian Motion',
        value: 1,
        itemStyle: {
          color: '#c89f83'
        }
      },
      {
        name: 'Multi-Pedestrian Motion',
        value: 1,
        itemStyle: {
          color: '#bb764c'
        }
      },
      {
        name: 'Traffic Signal',
        value: 1,
        itemStyle: {
          color: '#692a19'
        }
      },
      {
        name: 'Object Counting',
        value: 1,
        itemStyle: {
          color: '#470604'
        }
      },
      {
        name: 'Relationship (E2T)',
        value: 1,
        itemStyle: {
          color: '#CCCCE5'
        }
      },
      {
        name: 'Relationship (E2V)',
        value: 1,
        itemStyle: {
          color: '#E0E0FF'
        }
      },
      {
        name: 'Relationship (O2O)',
        value: 1,
        itemStyle: {
          color: '#CCCCFF'
        }
      },
      {
        name: 'Relationship (E2P)',
        value: 1,
        itemStyle: {
          color: '#8080FF'
        }
      },
      {
        name: 'Signal Attention',
        value: 1,
        itemStyle: {
          color: '#C0C0FF'
        }
      },
      {
        name: 'Ego Intention',
        value: 1,
        itemStyle: {
          color: '#CCCCFF'
        }
      },
      {
        name: 'Pedestrian Intention',
        value: 1,
        itemStyle: {
          color: '#C0C0FF'
        }
      },
      {
        name: 'Vehicle Intention',
        value: 1,
        itemStyle: {
          color: '#9999FF'
        }
      },
      ]
  },
  {
    name: 'Video \nMonitoring',
    // value: 1.1,
    itemStyle: {
      color: '#b09733'
    },
          children: [
      {
        name: 'Vehicle Counting',
        value: 1,
        itemStyle: {
          color: '#9db2b7'
        }
      },
      {
        name: 'Vehicle Existence',
        value: 1,
        itemStyle: {
          color: '#8b8c90'
        }
      },
  
      {
        name: 'Vehicle Location',
        value: 1,
        itemStyle: {
          color: '#beb276'
        }
      },
      {
        name: 'Pedestrian Counting',
        value: 1,
        itemStyle: {
          color: '#978847'
        }
      },
      {
        name: 'Pedestrian Existence',
        value: 1,
        itemStyle: {
          color: '#744e03'
        }
      },
      {
        name: 'Color Recognition',
        value: 1,
        itemStyle: {
          color: '#a3a36f'
        }
      },
      {
        name: 'Orientation Perception',
        value: 1,
        itemStyle: {
          color: '#c9b583'
        }
      },
      {
        name: 'Multi-Class Counting',
        value: 1,
        itemStyle: {
          color: '#E5E5E5'
        }
      },
      {
        name: 'Intention Prediction',
        value: 1,
        itemStyle: {
          color: '#E0E0E0'
        }
      },
      {
        name: 'Object Property',
        value: 1,
        itemStyle: {
          color: '#CCCCCC'
        }
      },
      ]
},
]
option = {
textStyle:{
fontStyle: 'normal',
fontFamily: "serif",
fontSize: 17,
width: 5,
overflow: 'breakAll'
},
series: {
type: 'sunburst',
data: data,
radius: ['15%', '80%'],
sort: undefined,
emphasis: {
  focus: 'ancestor'
},

levels: [
  
  {
  },
  {
    r0: '10%',
    r: '45%',
    itemStyle: {
      borderWidth: 2,
       borderRadius: 10,
      // opacity: 0.75,
      shadowColor: 'grey',
      shadowBlur: 20
    },
    label: {
      // rotate: 'tangential',
      align: 'center',
      fontSize: 12,
      fontWeight: "bold"
    }
  },
  {
    r0: '45%',
    r: '100%',
    label: {
      align: 'center',
      fontSize: 11,
      fontWeight: "bold"
    },
    itemStyle: {
      borderWidth: 3,
       borderRadius: 20,
       shadowColor: 'grey',
      // shadowBlur: 20
    },
  },
  {
    r0: '64%',
    r: '69%',
    label: {
      position: 'outside',
      padding: 0,
      fontSize: 14,
      silent: true,
      fontWeight: "bold"
    },
    itemStyle: {
      borderWidth: 2,
      borderRadius: 10,
      shadowColor: 'grey',
      
      // shadowBlur: 10
    }
  }
]
}
};

if (option && typeof option === 'object') {
  myChart.setOption(option);
}

window.addEventListener('resize', myChart.resize);
